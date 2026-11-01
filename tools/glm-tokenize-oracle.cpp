// CPU-only corpus token parity against the fixed independent GLM reference.
#include "llama.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio>
int main(int argc, char ** argv) {
    if (argc != 4) return 1;
    auto mp = llama_model_default_params();
    mp.vocab_only = true;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 2;
    const auto * vocab = llama_model_get_vocab(model);
    std::filesystem::path output(argv[3]);
    if (!std::filesystem::create_directory(output)) return 3;
    std::ifstream index(argv[2]); std::string line; int count = 0;
    while (std::getline(index, line)) {
        std::istringstream row(line); std::string mode, input, native;
        std::getline(row, mode, '\t'); std::getline(row, input, '\t'); std::getline(row, native);
        if (mode != "encode" && mode != "chat") return 4;
        std::ifstream in(input, std::ios::binary);
        if (!in) return 5;
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        bool special = mode == "chat";
        if (special) text = "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>" + text + "<|assistant|><think>";
        int n = -llama_tokenize(vocab, text.data(), int(text.size()), nullptr, 0, false, special);
        if (n < 0) return 6;
        std::vector<llama_token> tokens(n);
        n = llama_tokenize(vocab, text.data(), int(text.size()), tokens.data(), n, false, special);
        if (n < 0) return 7;
        auto path = output / std::filesystem::path(native).filename();
        if (std::filesystem::exists(path)) return 8;
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char *>(tokens.data()), n * sizeof(llama_token));
        out.flush(); if (!out) return 9;
        ++count;
    }
    llama_model_free(model);
    std::printf("tokenized_documents=%d\n", count); return 0;
}
