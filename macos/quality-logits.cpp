#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

// Standalone teacher-forced evaluator. Weights and generation are never changed.
int main(int argc, char ** argv) {
    if (argc != 5) return 1;
    const bool tokenize = std::string(argv[1]) == "tokenize";
    if (!tokenize && std::string(argv[1]) != "evaluate") return 1;
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.vocab_only = tokenize;
    mp.n_gpu_layers = tokenize ? 0 : 99;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    mp.lazy_mode = LLAMA_LAZY_MODE_ON;
    auto * model = llama_model_load_from_file(argv[2], mp);
    if (!model) { llama_backend_free(); return 2; }
    const auto * vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);
    int rc = 0;
    if (tokenize) {
        std::ifstream input(argv[3], std::ios::binary);
        std::string text((std::istreambuf_iterator<char>(input)), {});
        if (!input || text.empty()) rc = 3;
        else {
            int n = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, false, false);
            if (n >= 0) rc = 3;
            else {
                std::vector<llama_token> tokens(-n);
                n = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), false, false);
                if (n <= 0) rc = 3;
                else {
                    std::ofstream out(argv[4], std::ios::binary);
                    out.write(reinterpret_cast<const char *>(tokens.data()), n * sizeof(llama_token));
                    if (!out) rc = 3;
                    std::printf("{\"vocab\":%d,\"tokens\":%d,\"return\":%d}\n", nv, n, rc);
                }
            }
        }
    } else {
        auto cp = llama_context_default_params();
        cp.n_ctx = 8192;
        cp.n_batch = cp.n_ubatch = 1024;
        cp.expert_cache_slots = 64;
        cp.n_threads = cp.n_threads_batch = 8;
        auto * ctx = llama_init_from_model(model, cp);
        if (!ctx) rc = 2;
        else {
            std::ifstream manifest(argv[3]);
            std::string id, path;
            int documents = 0;
            while (manifest >> id >> path) {
                std::ifstream in(path, std::ios::binary | std::ios::ate);
                const auto bytes = in.tellg();
                if (!in || bytes <= 0 || bytes % sizeof(llama_token) != 0) { rc = 3; break; }
                std::vector<llama_token> tokens(bytes / sizeof(llama_token));
                in.seekg(0);
                in.read(reinterpret_cast<char *>(tokens.data()), bytes);
                if (!in || tokens.size() < 2 || tokens.size() > 1024) { rc = 3; break; }
                llama_memory_clear(llama_get_memory(ctx), true);
                auto batch = llama_batch_init(tokens.size(), 0, 1);
                batch.n_tokens = tokens.size();
                for (int i = 0; i < batch.n_tokens; ++i) {
                    if (tokens[i] < 0 || tokens[i] >= nv) { rc = 3; break; }
                    batch.token[i] = tokens[i];
                    batch.pos[i] = i;
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i] = true;
                }
                if (rc == 0) rc = llama_decode(ctx, batch);
                llama_synchronize(ctx);
                llama_batch_free(batch);
                if (rc != 0) break;
                const std::string base = std::string(argv[4]) + "/" + id;
                std::ofstream loss(base + ".loss.f64", std::ios::binary);
                std::ofstream samples(base + ".logits.f32", std::ios::binary);
                int eligible = 0, sampled = 0;
                for (size_t i = 0; i + 1 < tokens.size(); ++i) {
                    if (llama_vocab_get_attr(vocab, tokens[i + 1]) & LLAMA_TOKEN_ATTR_CONTROL) continue;
                    const float * logits = llama_get_logits_ith(ctx, i);
                    if (!logits) { rc = 4; break; }
                    double maximum = -INFINITY;
                    for (int j = 0; j < nv; ++j) {
                        if (!std::isfinite(logits[j])) { rc = 4; break; }
                        maximum = std::max(maximum, double(logits[j]));
                    }
                    if (rc != 0) break;
                    double sum = 0;
                    for (int j = 0; j < nv; ++j) sum += std::exp(double(logits[j]) - maximum);
                    const double nll = maximum + std::log(sum) - logits[tokens[i + 1]];
                    if (!std::isfinite(nll)) { rc = 4; break; }
                    loss.write(reinterpret_cast<const char *>(&nll), sizeof(nll));
                    if (eligible % 16 == 0) {
                        samples.write(reinterpret_cast<const char *>(logits), nv * sizeof(float));
                        ++sampled;
                    }
                    ++eligible;
                }
                if (!loss || !samples || eligible == 0) rc = 3;
                std::printf("{\"id\":\"%s\",\"tokens\":%zu,\"eligible\":%d,\"sampled\":%d,\"vocab\":%d,\"return\":%d}\n",
                            id.c_str(), tokens.size(), eligible, sampled, nv, rc);
                std::fflush(stdout);
                ++documents;
                if (rc != 0) break;
            }
            if (rc == 0 && (!manifest.eof() || documents == 0)) rc = 3;
            llama_free(ctx);
        }
    }
    llama_model_free(model);
    llama_backend_free();
    return rc;
}
