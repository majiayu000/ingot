// Fixed llama.cpp oracle only. Never linked into the Rust engine.
#include "llama.h"
#include "ggml-backend.h"
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
struct Trace{int position=0,target=-1;std::string dir;bool failed=false;};
static bool callback(ggml_tensor*t,bool ask,void*data){auto&x=*static_cast<Trace*>(data);std::string n=t->name;if(x.position!=x.target)return false;bool wanted=n.find("hc_mixed")==0||n.find("ffn_moe_topk")==0||n.find("ffn_moe_logits")==0||n.find("ffn_out")==0||n.find("l_last")==0||n.find("linear_attn_out")==0||n.find("attn_output")==0;if(!wanted)return false;if(ask)return true;if(!ggml_is_contiguous(t))return true;std::vector<char>bytes(ggml_nbytes(t));ggml_backend_tensor_get(t,bytes.data(),0,bytes.size());std::ofstream out(x.dir+"/"+n+".bin",std::ios::binary);out.write(bytes.data(),bytes.size());if(!out){x.failed=true;return false;}return true;}
int main(int argc,char**argv){
 if(argc!=5)return 1;
 llama_backend_init();ggml_backend_load_all();
 auto mp=llama_model_default_params();mp.n_gpu_layers=99;mp.load_mode=LLAMA_LOAD_MODE_MMAP;mp.lazy_mode=LLAMA_LAZY_MODE_ON;
 auto*model=llama_model_load_from_file(argv[1],mp);if(!model)return 2;
 auto cp=llama_context_default_params();cp.n_ctx=std::stoi(argv[4]);cp.n_batch=cp.n_ubatch=1;cp.n_threads=cp.n_threads_batch=8;cp.type_k=cp.type_v=GGML_TYPE_F32;cp.expert_cache_slots=64;
 Trace trace;if(const char*d=std::getenv("QWEN_ORACLE_TRACE_DIR")){trace.dir=d;std::filesystem::create_directories(trace.dir);const char*position=std::getenv("QWEN_ORACLE_TRACE_POSITION");if(!position){llama_model_free(model);llama_backend_free();return 1;}trace.target=std::stoi(position);cp.cb_eval=callback;cp.cb_eval_user_data=&trace;}
 if(std::getenv("QWEN_ORACLE_NO_FLASH"))cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
 auto*ctx=llama_init_from_model(model,cp);if(!ctx){llama_model_free(model);return 2;}
 std::ifstream in(argv[2],std::ios::binary);std::ofstream out(argv[3],std::ios::binary);int rc=0;int nv=llama_vocab_n_tokens(llama_model_get_vocab(model));llama_token token;int count=0;
 while(in.read(reinterpret_cast<char*>(&token),4)){trace.position=count;
  if(token<0||token>=nv||llama_decode(ctx,llama_batch_get_one(&token,1))){rc=3;break;}
  llama_synchronize(ctx);if(trace.failed){rc=5;break;}auto*l=llama_get_logits_ith(ctx,-1);if(!l){rc=4;break;}for(int i=0;i<nv;++i)if(!std::isfinite(l[i]))rc=4;if(rc)break;
  out.write(reinterpret_cast<const char*>(l),nv*4);if(!out){rc=5;break;}++count;
 }
 if(!in.eof()||in.gcount()!=0||count==0)rc=5;out.flush();if(!out)rc=5;
 llama_free(ctx);llama_model_free(model);llama_backend_free();std::printf("positions=%d vocab=%d return=%d\n",count,nv,rc);return rc;
}
