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
struct Measurement {
    size_t case_index;
    int iteration;
    bool warmup;
    std::vector<int> generated;
    std::vector<Clock::time_point> ready;
    Clock::time_point start, prefill, first, last;
};

template<class F>
auto invoke_forward(F f, Transformer* model, int token, int pos, bool logits, int)
    -> decltype(f(model, token, pos, logits)) { return f(model, token, pos, logits); }
template<class F>
auto invoke_forward(F f, Transformer* model, int token, int pos, bool, long)
    -> decltype(f(model, token, pos)) { return f(model, token, pos); }

// The mode is selected outside the timed region. Both instantiations use the
// same inference path and boundary stamps; only interior ITL stamps differ.
template<bool CollectITL>
void measure(Transformer& model, const Case& c, Measurement& r) {
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = Clock::now();
    float* logits = nullptr;
    for (int pos = 0; pos < int(c.ids.size()); ++pos)
        logits = invoke_forward(&forward, &model, c.ids[pos], pos,
                                pos + 1 == int(c.ids.size()), 0);
    // Current forward returns only after blocking D2H of its final logits.
    auto prefill = Clock::now();
    r.generated[0] = sample_argmax(logits);
    auto first = Clock::now();
    for (int i = 1; i < c.output - 1; ++i) {
        logits = invoke_forward(&forward, &model, r.generated[i-1],
                                int(c.ids.size())+i-1, true, 0);
        r.generated[i] = sample_argmax(logits);
        if constexpr (CollectITL) r.ready[i] = Clock::now();
    }
    // Share the final timestamp in both modes; no per-token timing-mode branch.
    logits = invoke_forward(&forward, &model, r.generated[c.output-2],
                            int(c.ids.size())+c.output-2, true, 0);
    r.generated[c.output-1] = sample_argmax(logits);
    auto last = Clock::now();
    // Checks and fallback synchronization are deliberately outside latency.
    // Any failure still invalidates the run; asynchronous candidates must make
    // host logits ready before returning through this forward interface.
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    r.start=start; r.prefill=prefill; r.first=first; r.last=last;
    if constexpr (CollectITL) { r.ready.front()=first; r.ready.back()=last; }
}

int main(int argc, char** argv) {
    try {
        if (argc != 5 && argc != 6)
            throw std::runtime_error("benchmark_probe MODEL_WEIGHTS PLAN WARMUP REPEATS [full|no-itl]");
        int warmup=std::stoi(argv[3]), repeats=std::stoi(argv[4]);
        std::string mode=argc==6 ? argv[5] : "full";
        if (mode!="full" && mode!="no-itl") throw std::runtime_error("Invalid timing mode");
        bool itl=mode=="full";
        if (warmup<0 || repeats<1) throw std::runtime_error("Invalid repetitions");
        std::ifstream plan(argv[2]);
        if (!plan) throw std::runtime_error("Cannot read plan");
        std::vector<Case> cases;
        std::string name,path; int output;
        while (plan >> name >> std::quoted(path) >> output) {
            std::ifstream input(path);
            if (!input) throw std::runtime_error("Cannot read input IDs");
            Case c{name,{},output}; int token;
            while (input >> token) {
                if (token<0 || token>=VOCAB_SIZE) throw std::runtime_error("Invalid token ID");
                c.ids.push_back(token);
            }
            if (!input.eof() || c.ids.empty() || output<2 || c.ids.size()+output>SEQ_LEN)
                throw std::runtime_error("Invalid workload");
            cases.push_back(c);
        }
        if (!plan.eof() || cases.empty()) throw std::runtime_error("Invalid plan");
        // Allocate/touch all recording buffers before any measured request.
        // Emit results only after the complete workload, so the Python reader
        // does not parse JSON or write progress/files alongside later requests.
        std::vector<Measurement> measurements;
        for (size_t k=0; k<cases.size(); ++k) {
            bool is_warmup=cases[k].name=="__warmup__";
            int count=is_warmup ? warmup : repeats;
            for (int i=0; i<count; ++i) {
                Measurement r{}; r.case_index=k; r.iteration=i; r.warmup=is_warmup;
                r.generated.resize(cases[k].output);
                r.ready.resize(cases[k].output); // Same allocation footprint in both modes.
                measurements.push_back(std::move(r));
            }
        }
        CUDA_CHECK(cudaFree(nullptr));
        CUDA_CHECK(cudaDeviceSynchronize());
        Transformer model{};
        auto load_start=Clock::now();
        build_transformer(&model,argv[1]);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        double load_ms=ms(load_start,Clock::now());
        for (auto& r:measurements) {
            if (itl) measure<true>(model,cases[r.case_index],r);
            else measure<false>(model,cases[r.case_index],r);
        }
        std::cout << std::setprecision(12);
        std::cout << "{\"event\":\"loaded\",\"protocol\":4,\"timing_mode\":\"" << mode
                  << "\",\"model_load_ms\":" << load_ms << "}\n";
        for (const auto& r:measurements) {
            const auto& c=cases[r.case_index];
            double decode=ms(r.first,r.last);
            std::cout << "{\"event\":\"request\",\"timing_protocol\":4,\"case\":\"" << c.name
                << "\",\"warmup\":" << (r.warmup ? "true" : "false")
                << ",\"iteration\":" << r.iteration
                << ",\"prompt_tokens\":" << c.ids.size() << ",\"output_tokens\":" << c.output
                << ",\"prefill_ms\":" << ms(r.start,r.prefill) << ",\"ttft_ms\":" << ms(r.start,r.first)
                << ",\"decode_ms\":" << decode << ",\"tpot_ms\":" << decode/(c.output-1)
                << ",\"decode_tokens_per_second\":" << 1000*(c.output-1)/decode
                << ",\"total_ms\":" << ms(r.start,r.last) << ",\"generated_ids\":[";
            for (int i=0;i<c.output;++i) std::cout << (i ? "," : "") << r.generated[i];
            std::cout << "],\"itl_enabled\":" << (itl ? "true" : "false") << ",\"itl_ms\":";
            if (itl) {
                std::cout << '[';
                for (int i=1;i<c.output;++i) std::cout << (i>1 ? "," : "") << ms(r.ready[i-1],r.ready[i]);
                std::cout << ']';
            } else std::cout << "null";
            // Retained for old readers: protocol 4 ends at token readiness.
            std::cout << ",\"decode_tail_ms\":0}\n";
        }
        free_transformer(&model);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
