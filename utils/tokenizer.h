#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <unicode/unorm2.h>
#include <unicode/ustring.h>
#include "config.h"

struct BpeMerge {
    uint32_t rank;
    int token;
};

struct Tokenizer {
    std::vector<std::string> vocab;
    std::unordered_map<uint64_t, BpeMerge> merges;
    std::vector<int> added_tokens;
    int byte_ids[256];
    bool normalize_nfc = false;
    unsigned int bos_token_id = 0;
    unsigned int eos_token_id = 0;
    char prompt_template[1024] = {};
    char system_prompt_template[1024] = {};
    std::unique_ptr<pcre2_code, decltype(&pcre2_code_free)> pattern{nullptr, pcre2_code_free};
};

inline uint64_t bpe_pair(int left, int right) {
    return (uint64_t(uint32_t(left)) << 32) | uint32_t(right);
}

inline uint32_t tokenizer_read_u32(std::istream& in) {
    unsigned char b[4];
    if (!in.read(reinterpret_cast<char*>(b), 4))
        throw std::runtime_error("Truncated tokenizer.bin; rerun tools/export.py");
    return uint32_t(b[0]) | (uint32_t(b[1]) << 8) | (uint32_t(b[2]) << 16) | (uint32_t(b[3]) << 24);
}

inline std::string tokenizer_read_string(std::istream& in, uint32_t length) {
    if (length > 1024*1024) throw std::runtime_error("Invalid tokenizer string length");
    std::string value(length, '\0');
    if (length && !in.read(&value[0], length)) throw std::runtime_error("Truncated tokenizer string");
    return value;
}

inline void load_single_template(char* buffer, size_t size, const char* dir, const char* name) {
    std::ifstream in(std::string(dir) + "/" + name, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("Cannot load template: ") + name);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (text.size() >= size) throw std::runtime_error("Prompt template is too long");
    std::memcpy(buffer, text.c_str(), text.size()+1);
}

inline void build_tokenizer(Tokenizer* t, const char* dir, int thinking) {
    std::ifstream in(std::string(dir) + "/tokenizer.bin", std::ios::binary);
    char magic[4];
    if (!in.read(magic, 4) || std::memcmp(magic, "QTK2", 4))
        throw std::runtime_error("Expected QTK2 tokenizer.bin; rerun: python tools/export.py <model_dir>");
    uint32_t count = tokenizer_read_u32(in);
    t->bos_token_id = tokenizer_read_u32(in);
    t->eos_token_id = tokenizer_read_u32(in);
    uint32_t merge_count = tokenizer_read_u32(in);
    uint32_t added_count = tokenizer_read_u32(in);
    uint32_t regex_length = tokenizer_read_u32(in);
    uint32_t normalization = tokenizer_read_u32(in);
    if (count == 0 || count > VOCAB_SIZE || added_count > count || merge_count > 1000000 ||
        regex_length == 0 || regex_length > 65536 || normalization > 1 ||
        t->bos_token_id >= count || t->eos_token_id >= count)
        throw std::runtime_error("Invalid tokenizer header");
    t->normalize_nfc = normalization != 0;
    std::string regex = tokenizer_read_string(in, regex_length);
    int error;
    PCRE2_SIZE offset;
    t->pattern.reset(pcre2_compile(reinterpret_cast<PCRE2_SPTR>(regex.data()), regex.size(),
                                   PCRE2_UTF | PCRE2_UCP, &error, &offset, nullptr));
    if (!t->pattern) throw std::runtime_error("Cannot compile tokenizer Unicode regex");
    t->vocab.assign(VOCAB_SIZE, std::string());
    for (uint32_t i = 0; i < count; ++i)
        t->vocab[i] = tokenizer_read_string(in, tokenizer_read_u32(in));
    for (int b = 0; b < 256; ++b) {
        uint32_t id = tokenizer_read_u32(in);
        if (id >= count || t->vocab[id] != std::string(1, char(b)))
            throw std::runtime_error("Invalid byte token mapping");
        t->byte_ids[b] = int(id);
    }
    t->added_tokens.clear();
    for (uint32_t i = 0; i < added_count; ++i) {
        uint32_t id = tokenizer_read_u32(in);
        if (id >= count || t->vocab[id].empty()) throw std::runtime_error("Invalid added token");
        t->added_tokens.push_back(int(id));
    }
    t->merges.clear();
    t->merges.reserve(merge_count);
    for (uint32_t rank = 0; rank < merge_count; ++rank) {
        uint32_t left = tokenizer_read_u32(in);
        uint32_t right = tokenizer_read_u32(in);
        uint32_t result = tokenizer_read_u32(in);
        if (left >= count || right >= count || result >= count ||
            t->vocab[left] + t->vocab[right] != t->vocab[result])
            throw std::runtime_error("Invalid BPE merge");
        if (!t->merges.emplace(bpe_pair(left, right), BpeMerge{rank, int(result)}).second)
            throw std::runtime_error("Duplicate BPE merge pair");
    }
    load_single_template(t->prompt_template, sizeof(t->prompt_template), dir,
                         thinking ? "template_user_thinking.txt" : "template_user.txt");
    load_single_template(t->system_prompt_template, sizeof(t->system_prompt_template), dir,
                         thinking ? "template_system_thinking.txt" : "template_system.txt");
}

inline void free_tokenizer(Tokenizer* t) {
    t->pattern.reset();
    t->vocab.clear();
    t->merges.clear();
    t->added_tokens.clear();
}

inline char* decode(Tokenizer* t, int token) {
    if (token < 0 || token >= int(t->vocab.size())) throw std::runtime_error("Invalid token ID");
    // Existing CLI treats this pointer as read-only.
    return const_cast<char*>(t->vocab[token].c_str());
}

inline void encode_bpe(Tokenizer* t, const char* text, size_t length, std::vector<int>& out) {
    std::vector<int> ids;
    ids.reserve(length);
    for (size_t i = 0; i < length; ++i) ids.push_back(t->byte_ids[static_cast<unsigned char>(text[i])]);
    while (ids.size() > 1) {
        uint32_t best_rank = std::numeric_limits<uint32_t>::max();
        size_t best_pos = 0;
        int result = -1;
        for (size_t i = 0; i+1 < ids.size(); ++i) {
            auto merge = t->merges.find(bpe_pair(ids[i], ids[i+1]));
            if (merge != t->merges.end() && merge->second.rank < best_rank) {
                best_rank = merge->second.rank;
                best_pos = i;
                result = merge->second.token;
            }
        }
        if (result == -1) break;
        ids[best_pos] = result;
        ids.erase(ids.begin() + best_pos + 1);
    }
    out.insert(out.end(), ids.begin(), ids.end());
}

inline std::string normalize_nfc(const std::string& input) {
    if (input.size() > size_t(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("Tokenizer input is too long");
    UErrorCode status = U_ZERO_ERROR;
    int32_t length = 0;
    u_strFromUTF8(nullptr, 0, &length, input.data(), int32_t(input.size()), &status);
    if (U_FAILURE(status) && status != U_BUFFER_OVERFLOW_ERROR)
        throw std::runtime_error("Tokenizer input must be valid UTF-8");
    status = U_ZERO_ERROR;
    std::vector<UChar> utf16(size_t(length)+1);
    u_strFromUTF8(utf16.data(), int32_t(utf16.size()), nullptr, input.data(), int32_t(input.size()), &status);
    if (U_FAILURE(status)) throw std::runtime_error("UTF-8 conversion failed");
    const UNormalizer2* nfc = unorm2_getNFCInstance(&status);
    if (U_FAILURE(status)) throw std::runtime_error("Cannot initialize NFC normalizer");
    int32_t normalized_length = unorm2_normalize(nfc, utf16.data(), length, nullptr, 0, &status);
    if (U_FAILURE(status) && status != U_BUFFER_OVERFLOW_ERROR)
        throw std::runtime_error("NFC normalization failed");
    status = U_ZERO_ERROR;
    std::vector<UChar> normalized(size_t(normalized_length)+1);
    unorm2_normalize(nfc, utf16.data(), length, normalized.data(), int32_t(normalized.size()), &status);
    if (U_FAILURE(status)) throw std::runtime_error("NFC normalization failed");
    int32_t bytes = 0;
    u_strToUTF8(nullptr, 0, &bytes, normalized.data(), normalized_length, &status);
    if (U_FAILURE(status) && status != U_BUFFER_OVERFLOW_ERROR)
        throw std::runtime_error("Normalized UTF-8 conversion failed");
    status = U_ZERO_ERROR;
    std::string output(size_t(bytes), '\0');
    if (bytes) u_strToUTF8(&output[0], bytes, nullptr, normalized.data(), normalized_length, &status);
    if (U_FAILURE(status)) throw std::runtime_error("Normalized UTF-8 conversion failed");
    return output;
}

inline void encode_plain(Tokenizer* t, const std::string& input, std::vector<int>& out) {
    const std::string text = t->normalize_nfc ? normalize_nfc(input) : input;
    std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)> match(
        pcre2_match_data_create_from_pattern(t->pattern.get(), nullptr), pcre2_match_data_free);
    if (!match) throw std::runtime_error("Cannot allocate tokenizer regex match data");
    size_t offset = 0;
    while (offset < text.size()) {
        int rc = pcre2_match(t->pattern.get(), reinterpret_cast<PCRE2_SPTR>(text.data()),
                            text.size(), offset, PCRE2_ANCHORED, match.get(), nullptr);
        if (rc < 0) throw std::runtime_error("Pre-tokenization failed; input must be valid UTF-8");
        PCRE2_SIZE* span = pcre2_get_ovector_pointer(match.get());
        if (span[0] != offset || span[1] <= offset) throw std::runtime_error("Invalid pre-token boundary");
        encode_bpe(t, text.data()+offset, span[1]-offset, out);
        offset = span[1];
    }
}

// Dynamic output is necessary: NFC can expand some Unicode characters.
inline std::vector<int> encode(Tokenizer* t, const std::string& input) {
    std::vector<int> out;
    size_t cursor = 0;
    while (cursor < input.size()) {
        size_t next = input.size();
        int special = -1;
        // Match only explicitly registered added tokens; '<' has no special scan logic.
        for (int id : t->added_tokens) {
            size_t found = input.find(t->vocab[id], cursor);
            if (found != std::string::npos && (found < next ||
                (found == next && (special == -1 || t->vocab[id].size() > t->vocab[special].size())))) {
                next = found;
                special = id;
            }
        }
        if (next > cursor) encode_plain(t, input.substr(cursor, next-cursor), out);
        if (special == -1) break;
        out.push_back(special);
        cursor = next + t->vocab[special].size();
    }
    return out;
}
