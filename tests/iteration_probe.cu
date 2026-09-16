// Model-boundary validation: no dependency on the names of individual kernels.
#include <cuda_bf16.h>
void iteration_trace(const __nv_bfloat16*, int, int);
#define QWEN_VALIDATION_TRACE iteration_trace
#include "models/qwen_model.cuh"
#include <fstream>
#include <iomanip>
#include <set>
#include <string>
#include <vector>
#include <stdexcept>

// Support both V0's three-argument forward and implementations with an optional
// logits flag. This adapter does not add any optimization to V0.
template<class F>
auto invoke_forward(F f, Transformer* model, int token, int pos, bool logits, int)
    -> decltype(f(model, token, pos, logits)) { return f(model, token, pos, logits); }
template<class F>
auto invoke_forward(F f, Transformer* model, int token, int pos, bool, long)
    -> decltype(f(model, token, pos)) { return f(model, token, pos); }

static std::ofstream* trace_stream = nullptr;
static std::set<int> trace_positions;
void iteration_trace(const bf16* x, int pos, int stage) {
    if (!trace_stream || !trace_positions.count(pos)) return;
    std::vector<bf16> values(DIM);
    CUDA_CHECK(cudaMemcpy(values.data(), x, DIM*sizeof(bf16), cudaMemcpyDeviceToHost));
    // Include coordinates: missing/reordered hooks must not silently compare as a different layer.
    trace_stream->write(reinterpret_cast<char*>(&pos), sizeof(pos));
    trace_stream->write(reinterpret_cast<char*>(&stage), sizeof(stage));
    for (auto value : values) {
        float f = __bfloat162float(value);
        trace_stream->write(reinterpret_cast<char*>(&f), sizeof(f));
    }
}
int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error("iteration_probe MODEL_WEIGHTS PLAN OUTPUT MODE(logits/trace)");
        bool trace = std::string(argv[4]) == "trace";
        if (!trace && std::string(argv[4]) != "logits") throw std::runtime_error("Invalid mode");
        std::ifstream plan(argv[2]);
        std::ofstream output(argv[3], std::ios::binary);
        if (!plan || !output) throw std::runtime_error("Cannot open plan/output");
        Transformer model{};
        build_transformer(&model, argv[1]);
        std::string path; int prompt, count;
        while (plan >> std::quoted(path) >> prompt >> count) {
            if (count < 1 || count > SEQ_LEN) throw std::runtime_error("Invalid snapshot count");
            std::set<int> positions;
            for (int i=0, pos; i<count; ++i) {
                if (!(plan >> pos) || pos < 0 || !positions.insert(pos).second)
                    throw std::runtime_error("Invalid positions");
            }
            std::ifstream input(path); std::vector<int> ids; int id;
            while (input >> id) {
                if (id < 0 || id >= VOCAB_SIZE) throw std::runtime_error("Invalid ID");
                ids.push_back(id);
            }
            if (!input.eof() || ids.empty() || ids.size()>SEQ_LEN || prompt<1 || prompt>int(ids.size()) || *positions.rbegin()>=int(ids.size()))
                throw std::runtime_error("Invalid history/prompt");
            // Reuse allocation, restart logical history. Past KV is overwritten causally.
            trace_positions=positions; trace_stream=trace ? &output : nullptr;
            for (int pos=0; pos<int(ids.size()); ++pos) {
                bool selected=positions.count(pos);
                // Decode consumes the fixed reference history, never the candidate argmax.
                bool need_logits=trace || selected || pos>=prompt-1;
                float* logits=invoke_forward(&forward,&model,ids[pos],pos,need_logits,0);
                CUDA_CHECK(cudaGetLastError());
                if (!trace && selected)
                    output.write(reinterpret_cast<char*>(logits), VOCAB_SIZE*sizeof(float));
            }
            trace_stream=nullptr;
        }
        if (!plan.eof()) throw std::runtime_error("Invalid plan");
        CUDA_CHECK(cudaDeviceSynchronize());
        if (!output) throw std::runtime_error("Output write failed");
        free_transformer(&model);
    } catch (const std::exception& e) { fprintf(stderr,"%s\n",e.what()); return 2; }
}
