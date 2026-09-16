// Test-only tracing is compiled out of the production CLI and benchmark.
#include <cuda_bf16.h>
void capture_validation_state(const __nv_bfloat16*, int, int);
#define QWEN_VALIDATION_TRACE capture_validation_state
#define main qwen600_cli_main
#include "engine/main.cu"
#undef main
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>

static std::ofstream* trace_output = nullptr;
static std::set<int> trace_positions;
void capture_validation_state(const __nv_bfloat16* x, int pos, int stage) {
    if (!trace_output || !trace_positions.count(pos)) return;
    std::vector<bf16> values(DIM);
    CUDA_CHECK(cudaMemcpy(values.data(), x, DIM*sizeof(bf16), cudaMemcpyDeviceToHost));
    for (auto value : values) {
        float f = __bfloat162float(value);
        trace_output->write(reinterpret_cast<char*>(&f), sizeof(f));
    }
}

constexpr int VALIDATION_MAX_TOKENS = SEQ_LEN;
static_assert(SEQ_LEN >= VALIDATION_MAX_TOKENS, "Validation exceeds allocated context");

static std::vector<int> read_ids(const char* path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open token file");
    std::vector<int> ids;
    int id;
    while (in >> id) {
        if (id < 0 || id >= VOCAB_SIZE) throw std::runtime_error("Invalid token ID");
        ids.push_back(id);
    }
    if (ids.empty() || ids.size() > VALIDATION_MAX_TOKENS)
        throw std::runtime_error("Validation input must contain 1..SEQ_LEN tokens");
    return ids;
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error("Expected tokenize, forward, greedy, or attention");
        std::string mode = argv[1];
        if (mode == "tokenize") {
            if (argc != 5) throw std::runtime_error("tokenize MODEL TEXT_FILE IDS_FILE");
            std::ifstream in(argv[3], std::ios::binary);
            if (!in) throw std::runtime_error("Cannot open text file");
            std::string text((std::istreambuf_iterator<char>(in)), {});
            Tokenizer t;
            build_tokenizer(&t, argv[2], 0);
            auto ids = encode(&t, text);
            std::ofstream out(argv[4]);
            if (!out) throw std::runtime_error("Cannot open token output");
            for (int id : ids) out << id << '\n';
            free_tokenizer(&t);
        } else if (mode == "attention") {
            // All Q/K entries are one, so every valid score must be sqrt(HEAD_DIM).
            bf16 *q, *k;
            float* att;
            CUDA_CHECK(cudaMalloc(&q, Q_DIM * sizeof(bf16)));
            CUDA_CHECK(cudaMalloc(&k, SEQ_LEN * KV_DIM * sizeof(bf16)));
            CUDA_CHECK(cudaMalloc(&att, N_HEADS * SEQ_LEN * sizeof(float)));
            std::vector<bf16> hq(Q_DIM, __float2bfloat16(1.f));
            std::vector<bf16> hk(SEQ_LEN * KV_DIM, __float2bfloat16(1.f));
            std::vector<float> ha(N_HEADS * SEQ_LEN, -1234.f);
            bool failed = false;
            for (int variant = 0; variant < 2; ++variant) {
                // Nonuniform, exactly representable values expose position/head indexing mistakes.
                if (variant) {
                    for (size_t i = 0; i < hq.size(); ++i)
                        hq[i] = __float2bfloat16(float(int((i * 17 + 3) % 31) - 15) / 16.f);
                    for (size_t i = 0; i < hk.size(); ++i)
                        hk[i] = __float2bfloat16(float(int((i * 13 + i / KV_DIM * 7) % 37) - 18) / 16.f);
                }
                CUDA_CHECK(cudaMemcpy(q, hq.data(), hq.size()*sizeof(bf16), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(k, hk.data(), hk.size()*sizeof(bf16), cudaMemcpyHostToDevice));
                std::vector<float> expected(N_HEADS * SEQ_LEN);
                for (int h = 0; h < N_HEADS; ++h)
                    for (int t = 0; t < SEQ_LEN; ++t) {
                        double sum = 0;
                        for (int i = 0; i < HEAD_DIM; ++i)
                            sum += double(__bfloat162float(hq[h*HEAD_DIM+i])) *
                                __bfloat162float(hk[t*KV_DIM+(h/(N_HEADS/N_KV_HEADS))*HEAD_DIM+i]);
                        expected[h*SEQ_LEN+t] = float(sum / sqrt(double(HEAD_DIM)));
                    }
                for (int pos : {0, 127, 511, 1022, 1023, 1024, 1025, 2047, 2048, 4095, 4096, SEQ_LEN-1}) {
                    std::fill(ha.begin(), ha.end(), -1234.f);
                    CUDA_CHECK(cudaMemcpy(att, ha.data(), ha.size()*sizeof(float), cudaMemcpyHostToDevice));
                    attention_qk_kernel<<<N_HEADS, std::min(1024, pos+1)>>>(att, q, k, pos);
                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaMemcpy(ha.data(), att, ha.size()*sizeof(float), cudaMemcpyDeviceToHost));
                    int bad = 0;
                    for (int h = 0; h < N_HEADS; ++h)
                        for (int t = 0; t < SEQ_LEN; ++t) {
                            float want = t <= pos ? expected[h*SEQ_LEN+t] : -1234.f;
                            if (!std::isfinite(ha[h*SEQ_LEN+t]) || fabsf(ha[h*SEQ_LEN+t] - want) > 1e-4f) ++bad;
                        }
                    std::cout << "{\"variant\":" << variant << ",\"position\":" << pos << ",\"incorrect_scores\":" << bad << "}\n";
                    failed |= bad != 0;
                }
            }
            CUDA_CHECK(cudaFree(q)); CUDA_CHECK(cudaFree(k)); CUDA_CHECK(cudaFree(att));
            return failed ? 1 : 0;
        } else if (mode == "forward" || mode == "greedy" || mode == "trace") {
            if (argc != 6) throw std::runtime_error("forward/greedy/trace MODEL IDS_FILE OUTPUT POSITIONS/COUNT");
            auto ids = read_ids(argv[3]);
            int generation_count = mode == "greedy" ? std::stoi(argv[5]) : 0;
            if (mode == "greedy" && (generation_count < 1 ||
                ids.size() + size_t(generation_count) > VALIDATION_MAX_TOKENS))
                throw std::runtime_error("Prompt plus generation must not exceed SEQ_LEN tokens");
            Transformer model;
            std::string weight_path = std::string(argv[2]) + "/model.safetensors";
            build_transformer(&model, weight_path.c_str());
            // Deterministic diagnostics: uncovered attention slots start as zero.
            CUDA_CHECK(cudaMemset(model.state.att, 0, N_HEADS*SEQ_LEN*sizeof(float)));
            std::ofstream out(argv[4], std::ios::binary);
            if (!out) throw std::runtime_error("Cannot open output");
            std::set<int> positions;
            if (mode == "forward" || mode == "trace") {
                std::istringstream stream(argv[5]);
                int pos;
                while (stream >> pos) {
                    if (pos < 0 || pos >= int(ids.size())) throw std::runtime_error("Invalid probe position");
                    positions.insert(pos);
                }
            }
            if (mode == "trace") { trace_output = &out; trace_positions = positions; }
            float* logits = nullptr;
            for (int pos = 0; pos < int(ids.size()); ++pos) {
                logits = forward(&model, ids[pos], pos);
                CUDA_CHECK(cudaGetLastError());
                if (mode == "forward" && positions.count(pos))
                    out.write(reinterpret_cast<char*>(logits), VOCAB_SIZE*sizeof(float));
            }
            trace_output = nullptr;
            if (mode == "greedy") {
                int count = generation_count;
                for (int i = 0; i < count; ++i) {
                    int token = sample_argmax(logits);
                    out << token << '\n';
                    if (token == 151645 || token == 151643) break;
                    if (i+1 < count) logits = forward(&model, token, int(ids.size())+i);
                }
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            free_transformer(&model);
        } else throw std::runtime_error("Unknown mode");
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 2;
    }
}
