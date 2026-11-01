// Independent fixed-library teacher forcing; validation only, no runtime linkage.
#include "llama.h"
#include "ggml-backend.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <filesystem>
int main(int argc,char**argv){
 if(argc!=4)return 1;llama_backend_init();ggml_backend_load_all();
 auto mp=llama_model_default_params();mp.n_gpu_layers=99;mp.load_mode=LLAMA_LOAD_MODE_MMAP;mp.lazy_mode=LLAMA_LAZY_MODE_ON;
 auto*m=llama_model_load_from_file(argv[1],mp);if(!m)return 2;
 auto cp=llama_context_default_params();cp.n_ctx=2048;cp.n_batch=cp.n_ubatch=1024;cp.n_threads=cp.n_threads_batch=8;cp.type_k=cp.type_v=GGML_TYPE_F32;cp.expert_cache_slots=0;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
 auto*ctx=llama_init_from_model(m,cp);if(!ctx)return 3;
 int nv=llama_vocab_n_tokens(llama_model_get_vocab(m));std::filesystem::path out(argv[3]);if(!std::filesystem::create_directory(out))return 4;
 std::ofstream scores(out/"scores.jsonl");std::ifstream index(argv[2]);std::string line;
 auto batch=llama_batch_init(1024,0,1);
 while(std::getline(index,line)){
  std::istringstream row(line);std::string cat,id,path,positions;std::getline(row,cat,'\t');std::getline(row,id,'\t');std::getline(row,path,'\t');std::getline(row,positions);
  std::set<int> samples;std::istringstream ps(positions);std::string part;while(std::getline(ps,part,','))samples.insert(std::stoi(part));
  std::ifstream input(path,std::ios::binary);std::vector<llama_token> tokens;llama_token t;while(input.read(reinterpret_cast<char*>(&t),4))tokens.push_back(t);
  if(!input.eof()||input.gcount()!=0||tokens.size()<2||tokens.size()>2048)return 5;for(auto t:tokens)if(t<0||t>=nv)return 5;
  std::ofstream full(out/(cat+"-"+id+".f32"),std::ios::binary);llama_memory_clear(llama_get_memory(ctx),true);double total=0;auto start=std::chrono::steady_clock::now();
  for(int pos=0;pos<int(tokens.size())-1;pos+=1024){
   int count=std::min(1024,int(tokens.size())-1-pos);batch.n_tokens=count;
   for(int i=0;i<count;i++){batch.token[i]=tokens[pos+i];batch.pos[i]=pos+i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=1;}
   if(llama_decode(ctx,batch))return 6;llama_synchronize(ctx);
   for(int i=0;i<count;i++){const float*l=llama_get_logits_ith(ctx,i);if(!l)return 7;double maximum=-INFINITY;for(int j=0;j<nv;j++){if(!std::isfinite(l[j]))return 8;maximum=std::max(maximum,double(l[j]));}double sum=0;for(int j=0;j<nv;j++)sum+=std::exp(double(l[j])-maximum);total+=maximum+std::log(sum)-l[tokens[pos+i+1]];if(samples.count(pos+i))full.write(reinterpret_cast<const char*>(l),nv*4);}
  }
  double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();scores.precision(17);scores<<"{\"category\":\""<<cat<<"\",\"id\":\""<<id<<"\",\"tokens\":"<<tokens.size()-1<<",\"nll_sum\":"<<total<<",\"seconds\":"<<seconds<<"}\n";scores.flush();std::printf("TF %s/%s tokens=%zu nll=%.9f seconds=%.3f\n",cat.c_str(),id.c_str(),tokens.size()-1,total,seconds);std::fflush(stdout);
 }
 llama_batch_free(batch);llama_free(ctx);llama_model_free(m);llama_backend_free();return 0;
}
