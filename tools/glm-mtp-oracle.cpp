// Validation-only GLM5-Next MTP oracle at 08ffba21e5c2f1728b6b405c05ff3913f65706d4, never linked into the native engine.
#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <cstdio>
#include <cstring>
#include <cmath>
struct Trace { std::filesystem::path dir; int pos=0; bool failed=false; };
static bool callback(ggml_tensor *t, bool ask, void *data) {
    auto &x=*static_cast<Trace*>(data); std::string name=t->name;
    bool wanted=name.find("mtp_")==0 || name.find("h_nextn")==0 || name.find("result_norm")==0 || name.find("result_output")==0;
    if(!wanted || !ggml_is_contiguous(t)) return false;
    if(ask) return true;
    auto dir=x.dir/std::to_string(x.pos);std::filesystem::create_directories(dir);
    std::vector<char> bytes(ggml_nbytes(t));ggml_backend_tensor_get(t,bytes.data(),0,bytes.size());
    std::ofstream out(dir/(name+".bin"),std::ios::binary);out.write(bytes.data(),bytes.size());
    if(!out) x.failed=true;
    return true;
}
int main(int argc,char **argv) {
    if(argc!=5) {std::fprintf(stderr,"oracle DRAFT TOKENS_U32 NATIVE_TRACE OUTPUT_DIR\n");return 1;}
    llama_backend_init();ggml_backend_load_all();
    auto mp=llama_model_default_params();mp.n_gpu_layers=99;mp.load_mtp=true;mp.load_mode=LLAMA_LOAD_MODE_MMAP;mp.lazy_mode=LLAMA_LAZY_MODE_ON;mp.use_extra_bufts=false;
    auto *model=llama_model_load_from_file(argv[1],mp);if(!model)return 2;
    Trace trace;trace.dir=argv[4];
    auto cp=llama_context_default_params();cp.ctx_type=LLAMA_CONTEXT_TYPE_MTP;cp.n_ctx=256;cp.n_batch=cp.n_ubatch=1;
    cp.n_threads=cp.n_threads_batch=8;cp.type_k=cp.type_v=GGML_TYPE_F32;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.moe_cache_size=size_t(8)*1024*1024*1024;cp.cb_eval=callback;cp.cb_eval_user_data=&trace;
    auto *ctx=llama_init_from_model(model,cp);if(!ctx){llama_model_free(model);return 3;}
    llama_set_embeddings_nextn(ctx,true,true);
    const int width=llama_model_n_embd_out(model);const int vocab=llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto batch=llama_batch_init(1,width,1);batch.token=static_cast<llama_token*>(std::malloc(sizeof(llama_token)));
    std::ifstream tokens(argv[2],std::ios::binary);llama_token token;int rc=0;
    while(tokens.read(reinterpret_cast<char*>(&token),4)) {
        if(trace.pos>=256 || token<0 || token>=vocab){rc=10;break;}
        std::ifstream hidden(std::filesystem::path(argv[3])/std::to_string(trace.pos)/"mtp_h_input-45.bin",std::ios::binary);
        if(!hidden.read(reinterpret_cast<char*>(batch.embd),width*4) || hidden.peek()!=std::char_traits<char>::eof()){rc=4;break;}
        for(int i=0;i<width;i++) if(!std::isfinite(batch.embd[i]))rc=4;
        if(rc)break;
        batch.n_tokens=1;batch.token[0]=token;batch.pos[0]=trace.pos;batch.n_seq_id[0]=1;batch.seq_id[0][0]=0;batch.logits[0]=1;
        if(llama_decode(ctx,batch)){rc=5;break;}llama_synchronize(ctx);
        if(trace.failed){rc=6;break;}
        const float *logits=llama_get_logits_ith(ctx,-1);if(!logits){rc=7;break;}
        int best=0;for(int i=0;i<vocab;i++){if(!std::isfinite(logits[i]))rc=8;if(logits[i]>logits[best])best=i;}
        std::printf("position=%d draft_argmax=%d\n",trace.pos,best);std::fflush(stdout);++trace.pos;
    }
    if(!tokens.eof() || tokens.gcount()!=0 || trace.pos==0)rc=9;
    std::free(batch.token);batch.token=nullptr;llama_batch_free(batch);llama_free(ctx);llama_model_free(model);llama_backend_free();return rc;
}
