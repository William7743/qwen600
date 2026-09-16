// Exercise the production implementation without changing its inference code.
#define main qwen600_cli_main
#include "engine/main.cu"
#undef main
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>

constexpr int VALIDATION_MAX_TOKENS = 1024;
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
        throw std::runtime_error("Validation input must contain 1..1024 tokens");
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
            CUDA_CHECK(cudaMemcpy(q, hq.data(), hq.size()*sizeof(bf16), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(k, hk.data(), hk.size()*sizeof(bf16), cudaMemcpyHostToDevice));
            bool failed = false;
            for (int pos : {0, 127, 511, 1022, 1023}) {
                std::fill(ha.begin(), ha.end(), -1234.f);
                CUDA_CHECK(cudaMemcpy(att, ha.data(), ha.size()*sizeof(float), cudaMemcpyHostToDevice));
                attention_qk_kernel<<<N_HEADS, std::min(1024, pos+1)>>>(att, q, k, pos);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(ha.data(), att, ha.size()*sizeof(float), cudaMemcpyDeviceToHost));
                int bad = 0;
                for (int h = 0; h < N_HEADS; ++h)
                    for (int t = 0; t <= pos; ++t)
                        if (fabsf(ha[h*SEQ_LEN+t] - sqrtf(float(HEAD_DIM))) > 1e-4f) ++bad;
                std::cout << "{\"position\":" << pos << ",\"incorrect_scores\":" << bad << "}\n";
                failed |= bad != 0;
            }
            CUDA_CHECK(cudaFree(q)); CUDA_CHECK(cudaFree(k)); CUDA_CHECK(cudaFree(att));
            return failed ? 1 : 0;
        } else if (mode == "forward" || mode == "greedy") {
            if (argc != 6) throw std::runtime_error("forward/greedy MODEL IDS_FILE OUTPUT POSITIONS/COUNT");
            auto ids = read_ids(argv[3]);
            int generation_count = mode == "greedy" ? std::stoi(argv[5]) : 0;
            if (mode == "greedy" && (generation_count < 1 ||
                ids.size() + size_t(generation_count) > VALIDATION_MAX_TOKENS))
                throw std::runtime_error("Prompt plus generation must not exceed 1024 tokens");
            Transformer model;
            std::string weight_path = std::string(argv[2]) + "/model.safetensors";
            build_transformer(&model, weight_path.c_str());
            // Deterministic diagnostics: uncovered attention slots start as zero.
            CUDA_CHECK(cudaMemset(model.state.att, 0, N_HEADS*SEQ_LEN*sizeof(float)));
            std::ofstream out(argv[4], std::ios::binary);
            if (!out) throw std::runtime_error("Cannot open output");
            std::set<int> positions;
            if (mode == "forward") {
                std::istringstream stream(argv[5]);
                int pos;
                while (stream >> pos) {
                    if (pos < 0 || pos >= int(ids.size())) throw std::runtime_error("Invalid probe position");
                    positions.insert(pos);
                }
            }
            float* logits = nullptr;
            for (int pos = 0; pos < int(ids.size()); ++pos) {
                logits = forward(&model, ids[pos], pos);
                CUDA_CHECK(cudaGetLastError());
                if (positions.count(pos))
                    out.write(reinterpret_cast<char*>(logits), VOCAB_SIZE*sizeof(float));
            }
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
