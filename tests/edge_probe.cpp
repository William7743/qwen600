// Run each case separately under AddressSanitizer so one failure cannot hide others.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <set>
#include "layers/sampler.h"
#include "utils/tokenizer.h"

void construct_path(char* out, size_t size, const char* dir, const char* file) {
    snprintf(out, size, "%s/%s", dir, file);
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    if (!strcmp(argv[1], "--build-info")) {
        bool asan = false;
#if defined(__SANITIZE_ADDRESS__)
        asan = true;
#endif
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
        asan = true;
#endif
#endif
        char pcre_version[64] = {};
        pcre2_config(PCRE2_CONFIG_VERSION, pcre_version);
        printf("{\"address_sanitizer\":%s,\"pcre2\":\"%s\",\"icu\":\"%s\"}\n",
               asan ? "true" : "false", pcre_version, U_ICU_VERSION);
        return 0;
    }
    if (!strcmp(argv[1], "tokenizer-less-than")) {
        if (argc != 3) return 2;
        Tokenizer t;
        build_tokenizer(&t, argv[2], 0);
        char* input = static_cast<char*>(malloc(2));
        input[0] = '<'; input[1] = 0;
        auto ids = encode(&t, input);
        if (ids.size() != 1 || t.vocab[ids[0]] != "<") return 1;
        free(input); free_tokenizer(&t);
        return 0;
    }
    std::vector<float> logits(VOCAB_SIZE, -100.f);
    logits[0] = logits[1] = 0.f;
    Sampler sampler;
    if (!strcmp(argv[1], "sampler-all")) {
        build_sampler(&sampler, 1.f, 0.95f, 0, 42);
        printf("token=%d\n", sample(&sampler, logits.data()));
    } else if (!strcmp(argv[1], "sampler-k-boundaries")) {
        std::fill(logits.begin(), logits.end(), -1000.f);
        logits[17] = 2.f; logits[999] = 3.f; logits[150000] = 4.f;
        for (int k : {0, 1, 2, 3, VOCAB_SIZE-1, VOCAB_SIZE, VOCAB_SIZE+1, -1}) {
            build_sampler(&sampler, 0.7f, 1.f, k, 42);
            for (int i = 0; i < 64; ++i) {
                int token = sample(&sampler, logits.data());
                if (token != 17 && token != 999 && token != 150000) return 1;
                if (k == 1 && token != 150000) return 1;
                if (k == 2 && token == 17) return 1;
            }
            free_sampler(&sampler);
        }
        puts("top-k boundary cases passed");
        return 0;
    } else if (!strcmp(argv[1], "sampler-nucleus-boundary")) {
        build_sampler(&sampler, 1.f, 0.5f, 2, 42);
        std::set<int> observed;
        for (int i = 0; i < 200; ++i) observed.insert(sample(&sampler, logits.data()));
        free_sampler(&sampler);
        puts("top-p=0.5 with equal candidates must retain one token");
        return observed.size() == 1 ? 0 : 1;
    } else if (!strcmp(argv[1], "sampler-distribution") ||
               !strcmp(argv[1], "sampler-distribution-default")) {
        // Two equal candidates should each receive probability 0.5 with top_p=1.
        float top_p = !strcmp(argv[1], "sampler-distribution-default") ? 0.95f : 1.f;
        build_sampler(&sampler, 1.f, top_p, 2, 42);
        int counts[2] = {0, 0};
        constexpr int runs = 2000;
        for (int i = 0; i < runs; ++i) {
            int token = sample(&sampler, logits.data());
            if (token < 0 || token > 1) return 1;
            ++counts[token];
        }
        printf("{\"draws\":%d,\"token0\":%d,\"token1\":%d,\"expected_each\":1000}\n",
               runs, counts[0], counts[1]);
        free_sampler(&sampler);
        // Generous 10% absolute tolerance for a deterministic diagnostic sample.
        return abs(counts[0]-runs/2) > runs/10 ? 1 : 0;
    } else return 2;
    free_sampler(&sampler);
    return 0;
}
