// Upstream llama.cpp logits oracle for the generic architectures (Llama,
// Qwen2/3, Qwen3-MoE). Never linked into the Rust engine.
// usage: generic-oracle MODEL TOKENS_FILE OUTPUT CONTEXT
// Feeds little-endian u32 tokens one at a time with an F32 KV cache and writes
// the F32 logits of every position.
#include "llama.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
int main(int argc, char **argv) {
    if (argc != 5) return 1;
    llama_backend_init();
    ggml_backend_load_all();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    auto *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 2;
    auto cp = llama_context_default_params();
    cp.n_ctx = std::stoi(argv[4]);
    cp.n_batch = cp.n_ubatch = 1;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto *ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); return 2; }
    std::ifstream in(argv[2], std::ios::binary);
    std::ofstream out(argv[3], std::ios::binary);
    int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model)), count = 0, rc = 0;
    llama_token token;
    while (in.read(reinterpret_cast<char *>(&token), 4)) {
        if (token < 0 || token >= vocab || llama_decode(ctx, llama_batch_get_one(&token, 1))) { rc = 3; break; }
        const float *l = llama_get_logits_ith(ctx, -1);
        if (!l) { rc = 4; break; }
        for (int i = 0; i < vocab; ++i) if (!std::isfinite(l[i])) rc = 4;
        if (rc) break;
        out.write(reinterpret_cast<const char *>(l), vocab * 4);
        ++count;
    }
    out.flush();
    if (!out || count == 0) rc = rc ? rc : 5;
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    std::printf("positions=%d vocab=%d return=%d\n", count, vocab, rc);
    return rc;
}
