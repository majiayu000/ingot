// GLM fixed 08ffba2 reference with expert-cache loading patch. Validation only.
// Fixed-library public generation for frozen official-answer tasks.
#include "llama.h"
#include "ggml-backend.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <filesystem>
int main(int argc,char**argv){
 if(argc!=5)return 1;if(std::filesystem::exists(argv[3]))return 2;
 llama_backend_init();ggml_backend_load_all();auto mp=llama_model_default_params();mp.n_gpu_layers=99;mp.load_mode=LLAMA_LOAD_MODE_NONE;mp.lazy_mode=LLAMA_LAZY_MODE_ON;mp.use_extra_bufts=false;
 auto*m=llama_model_load_from_file(argv[1],mp);if(!m)return 3;auto cp=llama_context_default_params();cp.n_ctx=std::stoul(argv[4]);cp.n_batch=cp.n_ubatch=64;cp.n_threads=cp.n_threads_batch=8;cp.type_k=cp.type_v=GGML_TYPE_F32;cp.moe_cache_size=size_t(16)*1024*1024*1024;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
 auto*ctx=llama_init_from_model(m,cp);if(!ctx)return 4;const auto*vocab=llama_model_get_vocab(m);int nv=llama_vocab_n_tokens(vocab);std::ofstream out(argv[3]);std::ifstream index(argv[2]);std::string line;
 while(std::getline(index,line)){
  std::istringstream row(line);std::string cat,id,path,capstr;std::getline(row,cat,'\t');std::getline(row,id,'\t');std::getline(row,path,'\t');std::getline(row,capstr);int cap=std::stoi(capstr);
  std::ifstream in(path,std::ios::binary);std::vector<llama_token> tokens;llama_token token;while(in.read(reinterpret_cast<char*>(&token),4))tokens.push_back(token);if(!in.eof()||in.gcount()!=0||tokens.empty()||tokens.size()+cap>cp.n_ctx)return 5;for(auto t:tokens)if(t<0||t>=nv)return 5;
  auto start=std::chrono::steady_clock::now();llama_memory_clear(llama_get_memory(ctx),true);
  for(size_t p=0;p<tokens.size();p+=64){int n=std::min<size_t>(64,tokens.size()-p);if(llama_decode(ctx,llama_batch_get_one(tokens.data()+p,n)))return 6;}
  llama_synchronize(ctx);double prefill=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();std::vector<int> output;const char*stop="Length";
  for(int i=0;i<cap;i++){const float*l=llama_get_logits_ith(ctx,-1);if(!l)return 7;int best=0;for(int j=0;j<nv;j++){if(!std::isfinite(l[j]))return 8;if(l[j]>l[best])best=j;}if(llama_vocab_is_eog(vocab,best)){stop="Eos";break;}output.push_back(best);token=best;if(i+1<cap&&llama_decode(ctx,llama_batch_get_one(&token,1)))return 9;}
  llama_synchronize(ctx);double total=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();out.precision(17);out<<"{\"category\":\""<<cat<<"\",\"id\":\""<<id<<"\",\"stop\":\""<<stop<<"\",\"tokens\":[";for(size_t i=0;i<output.size();i++)out<<(i?",":"")<<output[i];out<<"],\"total_s\":"<<total<<",\"prefill_s\":"<<prefill<<"}\n";out.flush();std::printf("TASK %s/%s %s output_tokens=%zu seconds=%.3f\n",cat.c_str(),id.c_str(),stop,output.size(),total);std::fflush(stdout);
 }
 llama_free(ctx);llama_model_free(m);llama_backend_free();return 0;
}
