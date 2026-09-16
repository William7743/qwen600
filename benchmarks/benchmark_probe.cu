#include "models/qwen_model.cuh"
#include "layers/sampler.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b-a).count();
}
struct Case { std::string name; std::vector<int> ids; int output; };

int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error("benchmark_probe MODEL_WEIGHTS PLAN WARMUP REPEATS");
        int warmup = std::stoi(argv[3]), repeats = std::stoi(argv[4]);
        if (warmup < 0 || repeats < 1) throw std::runtime_error("Invalid repetitions");
        std::ifstream plan(argv[2]);
        if (!plan) throw std::runtime_error("Cannot read plan");
        std::vector<Case> cases;
        std::string name, path;
        int output;
        while (plan >> name >> std::quoted(path) >> output) {
            std::ifstream input(path);
            if (!input) throw std::runtime_error("Cannot read input IDs");
            Case c{name, {}, output}; int token;
            while (input >> token) {
                if (token < 0 || token >= VOCAB_SIZE) throw std::runtime_error("Invalid token ID");
                c.ids.push_back(token);
            }
            if (!input.eof() || c.ids.empty() || output < 2 || c.ids.size()+output > SEQ_LEN)
                throw std::runtime_error("Invalid workload");
            cases.push_back(c);
        }
        if (!plan.eof() || cases.empty()) throw std::runtime_error("Invalid plan");
        CUDA_CHECK(cudaFree(nullptr)); // Initialize the context outside load timing.
        CUDA_CHECK(cudaDeviceSynchronize());
        Transformer model{};
        auto load_start = Clock::now();
        build_transformer(&model, argv[1]);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        double load_ms = ms(load_start, Clock::now());
        std::cout << std::setprecision(12);
        std::cout << "{\"event\":\"loaded\",\"model_load_ms\":" << load_ms << "}" << std::endl;
        for (const auto& c : cases) {
            for (int run = -warmup; run < repeats; ++run) {
                // Position restarts at zero. Every causal KV entry is overwritten before use.
                // Output storage is allocated before the timed interval.
                std::vector<int> generated(c.output);
                CUDA_CHECK(cudaDeviceSynchronize());
                auto t0 = Clock::now();
                float* logits = nullptr;
                for (int pos = 0; pos < int(c.ids.size()); ++pos)
                    logits = forward(&model, c.ids[pos], pos);
                CUDA_CHECK(cudaGetLastError());
                // forward's blocking device-to-host copy has made logits available.
                auto t1 = Clock::now();
                generated[0] = sample_argmax(logits);
                auto t2 = Clock::now();
                for (int i = 1; i < c.output; ++i) {
                    logits = forward(&model, generated[i-1], int(c.ids.size())+i-1);
                    generated[i] = sample_argmax(logits);
                }
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                auto t3 = Clock::now();
                // No EOS termination: exact work budget, independent of generated content.
                double decode = ms(t2,t3);
                std::cout << "{\"event\":\"request\",\"case\":\"" << c.name
                    << "\",\"warmup\":" << (run < 0 ? "true" : "false")
                    << ",\"iteration\":" << (run < 0 ? run+warmup : run)
                    << ",\"prompt_tokens\":" << c.ids.size() << ",\"output_tokens\":" << c.output
                    << ",\"prefill_ms\":" << ms(t0,t1) << ",\"ttft_ms\":" << ms(t0,t2)
                    << ",\"decode_ms\":" << decode << ",\"tpot_ms\":" << decode/(c.output-1)
                    << ",\"decode_tokens_per_second\":" << 1000*(c.output-1)/decode
                    << ",\"total_ms\":" << ms(t0,t3) << ",\"generated_ids\":[";
                for (int i=0; i<c.output; ++i) std::cout << (i ? "," : "") << generated[i];
                std::cout << "]}" << std::endl;
            }
        }
        free_transformer(&model);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
