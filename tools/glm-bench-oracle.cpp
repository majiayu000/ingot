// Same-token GLM fixed08ff AR baseline, expert byte cache. Measurement only.
#include "llama.h"
#include "ggml-backend.h"
#include <chrono>
#include <fstream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point p){return std::chrono::duration<double>(Clock::now()-p).count();}
int main(int argc,char**argv){
 if(argc!=8){std::fprintf(stderr,"bench MODEL TOKENS_U32 CONTEXT CAP REPEATS EXPERT_CACHE_GIB BATCH\n");return 1;}
 auto load=Clock::now();llama_backend_init();ggml_backend_load_all();
 auto mp=llama_model_default_params();mp.n_gpu_layers=99;mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.lazy_mode=LLAMA_LAZY_MODE_ON;mp.use_extra_bufts=false;
 auto*m=llama_model_load_from_file(argv[1],mp);if(!m)return 2;
 auto cp=llama_context_default_params();cp.n_ctx=std::atoi(argv[3]);cp.n_batch=cp.n_ubatch=std::atoi(argv[7]);cp.n_threads=cp.n_threads_batch=8;
 cp.type_k=cp.type_v=GGML_TYPE_F32;cp.moe_cache_size=size_t(std::stoul(argv[6]))*1024*1024*1024;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
 if(cp.n_batch<1 || cp.n_batch>1024)return 9;
 auto*ctx=llama_init_from_model(m,cp);if(!ctx){llama_model_free(m);return 3;}
 double load_s=elapsed(load);std::ifstream in(argv[2],std::ios::binary);std::vector<llama_token> prompt;llama_token token;
 while(in.read(reinterpret_cast<char*>(&token),4))prompt.push_back(token);
 const auto*vocab=llama_model_get_vocab(m);int nv=llama_vocab_n_tokens(vocab),cap=std::atoi(argv[4]),repeats=std::atoi(argv[5]);
 if(!in.eof()||in.gcount()!=0||prompt.empty()||prompt.size()+cap>cp.n_ctx)return 4;
 for(auto id:prompt)if(id<0||id>=nv)return 4;
 for(int round=0;round<=repeats;round++){
  auto start=Clock::now();llama_memory_clear(llama_get_memory(ctx),true);
  for(size_t pos=0;pos<prompt.size();pos+=cp.n_batch){int n=std::min<size_t>(cp.n_batch,prompt.size()-pos);if(llama_decode(ctx,llama_batch_get_one(prompt.data()+pos,n)))return 5;}
  llama_synchronize(ctx);double prefill=elapsed(start),first=0,visible=0;std::vector<int> output;const char*stop="Length";
  for(int i=0;i<cap;i++){
   const float*l=llama_get_logits_ith(ctx,-1);if(!l)return 6;int best=0;
   for(int j=0;j<nv;j++){if(!std::isfinite(l[j]))return 7;if(l[j]>l[best])best=j;}
   if(i==0)first=elapsed(start);if(llama_vocab_is_eog(vocab,best)){stop="Eos";break;}
   output.push_back(best);char piece[1024];int bytes=llama_token_to_piece(vocab,best,piece,sizeof(piece),0,false);if(!visible && bytes!=0)visible=elapsed(start);
   token=best;if(i+1<cap){if(llama_decode(ctx,llama_batch_get_one(&token,1)))return 8;llama_synchronize(ctx);}
  }
  double total=elapsed(start);
  std::printf("{\"round\":%d,\"load_s\":%.9f,\"prefill_s\":%.9f,\"first_token_s\":%.9f,\"first_visible_s\":%.9f,\"total_s\":%.9f,\"stop\":\"%s\",\"tokens\":[",round,load_s,prefill,first,visible,total,stop);
  for(size_t i=0;i<output.size();i++)std::printf("%s%d",i?",":"",output[i]);std::printf("]}\n");std::fflush(stdout);
 }
 llama_free(ctx);llama_model_free(m);llama_backend_free();return 0;
}
