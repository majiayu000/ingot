// Fixed llama.cpp mtmd oracle for image input. Never linked into the Rust engine.
//   embed    MODEL MMPROJ IMAGE OUT.f32         merged image embeddings (vocab-only model)
//   generate MODEL MMPROJ PROMPT N OUT.tokens IMAGE...
//            PROMPT is a file of raw chat text with one <__media__> per image; prints
//            the chunk layout and writes N greedy token ids (int32, stops at EOG).
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "gguf.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static mtmd_bitmap*load_image(mtmd_context*ctx,const char*path){auto w=mtmd_helper_bitmap_init_from_file(ctx,path,false,mtmd_helper_init_opt_default());return w.bitmap;}

// A vocab-only model reports no embedding width; the projector records it.
static int projection_dim(const char*path){gguf_init_params p{true,nullptr};auto*g=gguf_init_from_file(path,p);if(!g)return 0;auto k=gguf_find_key(g,"clip.vision.projection_dim");int d=k<0?0:(int)gguf_get_val_u32(g,k);gguf_free(g);return d;}

int main(int argc,char**argv){
 if(argc<6)return 1;std::string mode=argv[1];
 llama_backend_init();ggml_backend_load_all();
 auto mp=llama_model_default_params();mp.n_gpu_layers=99;mp.load_mode=LLAMA_LOAD_MODE_MMAP;mp.lazy_mode=LLAMA_LAZY_MODE_ON;mp.vocab_only=mode=="embed";
 auto*model=llama_model_load_from_file(argv[2],mp);if(!model)return 2;
 auto mparams=mtmd_context_params_default();mparams.warmup=false;mparams.print_timings=false;
 auto*mctx=mtmd_init_from_file(argv[3],model,mparams);if(!mctx){llama_model_free(model);return 2;}
 int rc=0;
 if(mode=="embed"){
  auto*bitmap=load_image(mctx,argv[4]);if(!bitmap)return 3;
  std::string text=mtmd_get_marker(mctx);mtmd_input_text input{text.c_str(),text.size(),false,true};
  auto*chunks=mtmd_input_chunks_init();const mtmd_bitmap*bitmaps[]={bitmap};
  if(mtmd_tokenize(mctx,chunks,&input,bitmaps,1))return 4;
  for(size_t i=0;i<mtmd_input_chunks_size(chunks);++i){auto*c=mtmd_input_chunks_get(chunks,i);if(mtmd_input_chunk_get_type(c)!=MTMD_INPUT_CHUNK_TYPE_IMAGE)continue;
   if(mtmd_encode_chunk(mctx,c))return 5;size_t n=mtmd_input_chunk_get_n_tokens(c);int d=projection_dim(argv[3]);
   std::ofstream out(argv[5],std::ios::binary);out.write(reinterpret_cast<const char*>(mtmd_get_output_embd(mctx)),n*d*4);
   std::printf("tokens=%zu n_pos=%d embd=%d\n",n,mtmd_input_chunk_get_n_pos(c),d);}
  mtmd_input_chunks_free(chunks);mtmd_bitmap_free(bitmap);
 }else if(mode=="generate"&&argc>=7){
  std::ifstream pf(argv[4]);std::stringstream ss;ss<<pf.rdbuf();std::string text=ss.str();std::string marker=mtmd_get_marker(mctx);for(size_t at=0;(at=text.find("<__media__>",at))!=std::string::npos;at+=marker.size())text.replace(at,11,marker);int n_gen=std::stoi(argv[5]);
  std::vector<mtmd_bitmap*>bitmaps;for(int i=7;i<argc;++i){auto*b=load_image(mctx,argv[i]);if(!b)return 3;bitmaps.push_back(b);}
  auto cp=llama_context_default_params();cp.n_ctx=16384;cp.n_batch=cp.n_ubatch=512;cp.n_threads=cp.n_threads_batch=8;cp.type_k=cp.type_v=GGML_TYPE_F32;
  auto*ctx=llama_init_from_model(model,cp);if(!ctx)return 2;
  mtmd_input_text input{text.c_str(),text.size(),false,true};auto*chunks=mtmd_input_chunks_init();
  if(mtmd_tokenize(mctx,chunks,&input,const_cast<const mtmd_bitmap**>(bitmaps.data()),bitmaps.size()))return 4;
  for(size_t i=0;i<mtmd_input_chunks_size(chunks);++i){auto*c=mtmd_input_chunks_get(chunks,i);
   if(mtmd_input_chunk_get_type(c)==MTMD_INPUT_CHUNK_TYPE_TEXT){size_t n;auto*t=mtmd_input_chunk_get_tokens_text(c,&n);std::printf("text");for(size_t j=0;j<n;++j)std::printf(" %d",t[j]);std::printf("\n");}
   else std::printf("image tokens=%zu n_pos=%d\n",mtmd_input_chunk_get_n_tokens(c),mtmd_input_chunk_get_n_pos(c));}
  llama_pos past=0;if(mtmd_helper_eval_chunks(mctx,ctx,chunks,0,0,512,true,&past))return 5;
  auto*vocab=llama_model_get_vocab(model);int nv=llama_vocab_n_tokens(vocab);std::ofstream out(argv[6],std::ios::binary);std::printf("generated");
  for(int i=0;i<n_gen;++i){auto*l=llama_get_logits_ith(ctx,-1);int best=0;for(int j=1;j<nv;++j)if(l[j]>l[best])best=j;
   std::printf(" %d",best);std::fflush(stdout);out.write(reinterpret_cast<const char*>(&best),4);if(llama_vocab_is_eog(vocab,best))break;
   llama_token t=best;llama_batch b=llama_batch_get_one(&t,1);llama_pos p=past++;b.pos=&p;if(llama_decode(ctx,b)){rc=6;break;}}
  std::printf("\n");mtmd_input_chunks_free(chunks);for(auto*b:bitmaps)mtmd_bitmap_free(b);llama_free(ctx);
 }else rc=1;
 mtmd_free(mctx);llama_model_free(model);llama_backend_free();return rc;
}
