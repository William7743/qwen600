// Length-prefixed corpus input/output, including embedded NUL and empty strings.
#include "utils/tokenizer.h"

static void write_u32(std::ostream& out, uint32_t value) {
    char bytes[4];
    for (int i = 0; i < 4; ++i) bytes[i] = char(value >> (8*i));
    out.write(bytes, 4);
}

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("tokenizer_probe MODEL INPUT OUTPUT");
        Tokenizer t;
        build_tokenizer(&t, argv[1], 0);
        std::ifstream in(argv[2], std::ios::binary);
        std::ofstream out(argv[3], std::ios::binary);
        if (!in || !out) throw std::runtime_error("Cannot open corpus files");
        uint32_t count = tokenizer_read_u32(in);
        write_u32(out, count);
        for (uint32_t i = 0; i < count; ++i) {
            auto text = tokenizer_read_string(in, tokenizer_read_u32(in));
            auto ids = encode(&t, text);
            write_u32(out, uint32_t(ids.size()));
            for (int id : ids) write_u32(out, uint32_t(id));
        }
        free_tokenizer(&t);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
