// Validation-only oracle, pinned to ggml-org/llama.cpp in PROVENANCE.md.
#include "llama.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

struct Trace {
    int position = 0;
    int target = -1;
    std::string directory;
    bool failed = false;
};
static bool callback(ggml_tensor * tensor, bool ask, void * opaque) {
    auto & trace = *static_cast<Trace *>(opaque);
    if (trace.position != trace.target) return false;
    std::string name = tensor->name;
    const char * prefixes[] = {"attn_norm", "attn_out", "ffn_norm", "ffn_out", "l_out",
        "kda_scan_out", "kda_out", "kv_cmpr", "q_absorbed", "indexer_sel", "ffn_moe_topk", "ffn_moe_logits", "h_nextn", "result_norm"};
    bool wanted = false;
    for (const char * prefix : prefixes) wanted |= name.rfind(prefix, 0) == 0;
    if (!wanted) return false;
    if (ask) return true;
    if (!ggml_is_contiguous(tensor)) return true;
    std::vector<char> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    std::ofstream out(trace.directory + "/" + name + ".bin", std::ios::binary);
    out.write(bytes.data(), bytes.size());
    trace.failed |= !out;
    return true;
}
int main(int argc, char ** argv) {
    const bool tokenize = argc >= 2 && std::string(argv[1]) == "tokenize";
    if ((!tokenize && argc != 5) || (tokenize && argc != 4 && argc != 5)) return 1;
    llama_backend_init();
    ggml_backend_load_all();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    // tools/glm-oracle-loading.patch enables host mappings only for routed experts.
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.lazy_mode = LLAMA_LAZY_MODE_ON;
    mp.use_extra_bufts = false;
    mp.vocab_only = tokenize;
    auto * model = llama_model_load_from_file(argv[tokenize ? 2 : 1], mp);
    if (!model) { llama_backend_free(); return 2; }
    const auto * vocab = llama_model_get_vocab(model);
    if (tokenize) {
        const std::string text = argv[3];
        const bool special = argc == 5 && std::string(argv[4]) == "--special";
        int count = -llama_tokenize(vocab, text.data(), int(text.size()), nullptr, 0, false, special);
        std::vector<llama_token> ids(count);
        count = llama_tokenize(vocab, text.data(), int(text.size()), ids.data(), count, false, special);
        int rc = 0;
        if (count < 0) rc = 3;
        else {
            std::printf("[");
            for (int i = 0; i < count; ++i) std::printf("%s%d", i ? ", " : "", ids[i]);
            std::printf("]\n");
        }
        llama_model_free(model); llama_backend_free(); return rc;
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = std::stoi(argv[4]);
    cp.n_batch = cp.n_ubatch = 1;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.moe_cache_size = size_t(4) * 1024 * 1024 * 1024;
    Trace trace;
    if (const char * dir = std::getenv("GLM_ORACLE_TRACE_DIR")) {
        trace.directory = dir;
        std::filesystem::create_directories(dir);
        const char * pos = std::getenv("GLM_ORACLE_TRACE_POSITION");
        if (!pos) { llama_model_free(model); llama_backend_free(); return 1; }
        trace.target = std::stoi(pos);
        cp.cb_eval = callback; cp.cb_eval_user_data = &trace;
    }
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); llama_backend_free(); return 2; }
    std::ifstream in(argv[2], std::ios::binary);
    std::ofstream out(argv[3], std::ios::binary);
    int rc = 0, count = 0;
    const int nv = llama_vocab_n_tokens(vocab);
    llama_token token;
    while (in.read(reinterpret_cast<char *>(&token), 4)) {
        trace.position = count;
        if (token < 0 || token >= nv || llama_decode(ctx, llama_batch_get_one(&token, 1))) { rc = 3; break; }
        llama_synchronize(ctx);
        if (trace.failed) { rc = 5; break; }
        auto * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) { rc = 4; break; }
        for (int i = 0; i < nv; ++i) if (!std::isfinite(logits[i])) rc = 4;
        if (rc) break;
        out.write(reinterpret_cast<const char *>(logits), nv * 4);
        if (!out) { rc = 5; break; }
        ++count;
    }
    if (!rc && (!in.eof() || in.gcount() != 0 || count == 0)) rc = 5;
    out.flush(); if (!rc && !out) rc = 5;
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    std::printf("positions=%d vocab=%d return=%d\n", count, nv, rc);
    return rc;
}
