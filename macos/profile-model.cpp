#include "llama.h"
#include "ggml-backend.h"
#include <chrono>
#include <cstdio>
#include <map>
#include <string>

using clock_type = std::chrono::steady_clock;
struct timing { double us = 0; int count = 0; };
struct profile {
    bool record = false;
    clock_type::time_point start;
    std::map<std::string, timing> totals;
};

static bool callback(ggml_tensor * t, bool ask, void * data) {
    auto & p = *static_cast<profile *>(data);
    if (!p.record || t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW ||
        t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE) {
        return false;
    }
    if (ask) {
        p.start = clock_type::now();
    } else {
        const double us = std::chrono::duration<double, std::micro>(clock_type::now() - p.start).count();
        std::string key = ggml_op_desc(t);
        if (t->src[0]) {
            key += std::string("/") + ggml_type_name(t->src[0]->type) + "/" +
                std::to_string(t->src[0]->ne[0]) + "x" + std::to_string(t->src[0]->ne[1]);
        }
        auto & total = p.totals[key];
        total.us += us;
        total.count++;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s model.gguf\n", argv[0]);
        return 1;
    }
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    mp.lazy_mode = LLAMA_LAZY_MODE_ON;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { llama_backend_free(); return 1; }
    profile p;
    auto cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = 4;
    cp.n_ubatch = 4;
    cp.n_threads = cp.n_threads_batch = 8;
    cp.cb_eval = callback;
    cp.cb_eval_user_data = &p;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); llama_backend_free(); return 1; }
    llama_token token = 1;
    int result = 0;
    for (int i = 0; i < 5; ++i) {
        p.record = i >= 2;
        if (llama_decode(ctx, llama_batch_get_one(&token, 1)) != 0) {
            result = 1;
            break;
        }
        llama_synchronize(ctx);
    }
    if (!result) {
        std::puts("operator,calls,total_us,mean_us");
        for (const auto & [key, value] : p.totals) {
            std::printf("%s,%d,%.3f,%.3f\n", key.c_str(), value.count, value.us, value.us/value.count);
        }
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return result;
}
