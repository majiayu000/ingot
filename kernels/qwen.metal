// F32 model operators. Weight layouts/codebooks are documented in NOTICE.
// [token, row] output. Each SIMD group reuses one quantized row for four
// tokens. Each token keeps the GEMV accumulation/reduction order unchanged.
kernel void quant_matmul(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
 threadgroup float iq4_lut[16];
 if(projection_type==23 && p[5]==4) {
  if(simd==0&&lane<16)iq4_lut[lane]=float(kvalues_iq4nl[lane]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint row_id=g.x*4+simd;if(row_id>=p[1])return;uint first=g.y*4;
  device const uchar*row=w+(ulong(p[4])*p[1]+row_id)*p[2];float4 s;
  if(p[5]==1)s=dot_iq4_shared(row,x,p[0],lane,0,1,iq4_lut);
  else if(first+4<=p[5])s=dot_iq4_shared(row,x+ulong(first)*p[0],p[0],lane,p[0],4,iq4_lut);
  else s=dot_iq4_shared(row,x+ulong(first)*p[0],p[0],lane,p[0],p[5]-first,iq4_lut);
  s.x=simd_sum(s.x);s.y=simd_sum(s.y);s.z=simd_sum(s.z);s.w=simd_sum(s.w);
  if(!lane)for(uint t=0;t<min(4u,p[5]-first);++t)y[ulong(first+t)*p[1]+row_id]=s[t];
  return;
 }
 uint row_id=g.x*4+simd;if(row_id>=p[1])return;
 // Specialize the single-token tail: no dynamic float4 accumulator or input branches.
 if(p[5]==1) {
  device const uchar*row=w+(ulong(p[4])*p[1]+row_id)*p[2];
  float value=simd_sum(dot_quant_row(row,x,p[0],lane));
  if(!lane)y[row_id]=value;
  return;
 }
 uint first=g.y*4;
 device const uchar*row=w+(ulong(p[4])*p[1]+row_id)*p[2];
 float4 s;
 if(projection_type!=29 && first+4<=p[5])s=dot_quant_rows(row,x+ulong(first)*p[0],p[0],lane,p[0],4);
 else s=dot_quant_rows(row,x+ulong(first)*p[0],p[0],lane,p[0],p[5]-first);
 s.x=simd_sum(s.x);s.y=simd_sum(s.y);s.z=simd_sum(s.z);s.w=simd_sum(s.w);
 if(!lane){y[ulong(first)*p[1]+row_id]=s.x;if(first+1<p[5])y[ulong(first+1)*p[1]+row_id]=s.y;if(first+2<p[5])y[ulong(first+2)*p[1]+row_id]=s.z;if(first+3<p[5])y[ulong(first+3)*p[1]+row_id]=s.w;}
}
inline float sigmoid_f(float x){return 1.0f/(1.0f+exp(-x));}
kernel void read_rows(device const uchar*w [[buffer(0)]],device float*y [[buffer(1)]],constant uint*p [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<p[0]*p[1])y[i]=weight_at(w+ulong(i/p[0])*p[2],p[3],i%p[0]);}
kernel void rms(device const float*x [[buffer(0)]],device const float*w [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
 uint n=p[0],offset=p[2],stride=p[3];float sum=0;for(uint j=lane;j<n;j+=32){float v=x[offset+row*stride+j];sum+=v*v;}sum=simd_sum(sum);float inv=rsqrt(sum/(p[5]?1.0f:float(n))+as_type<float>(p[4]));for(uint j=lane;j<n;j+=32)y[row*n+j]=x[offset+row*stride+j]*inv*(p[5]?1.0f:w[(row*n+j)%p[1]]);
}
kernel void unary(device const float*x [[buffer(0)]],device float*y [[buffer(1)]],constant uint*p [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i>=p[0])return;float v=x[i]*as_type<float>(p[2]);y[i]=p[1]==0?sigmoid_f(v):v*sigmoid_f(v);}
kernel void hc_collapse(device const float*x [[buffer(0)]],device const float*g [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint i [[thread_position_in_grid]]){if(i>=p[0])return;float v=0;for(uint h=0;h<p[1];++h)v+=x[h*p[0]+i]*g[h*p[0]+i];y[i]=v/p[1];}
kernel void hc_add(device const float*x [[buffer(0)]],device const float*b [[buffer(1)]],device const float*g [[buffer(2)]],device float*y [[buffer(3)]],constant uint*p [[buffer(4)]],uint i [[thread_position_in_grid]]){if(i<p[0]*p[1])y[i]=x[i]+b[i%p[0]]*2.0f*sigmoid_f(g[i/p[0]]/p[1]);}
kernel void repeat_emb(device const float*x [[buffer(0)]],device float*y [[buffer(1)]],constant uint*p [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<p[0]*p[1])y[i]=x[i%p[0]];}
// One SIMD group selects top-k; ties retain the lowest expert ID.
kernel void route_gpu(device const float*l [[buffer(0)]],device uint*ids [[buffer(1)]],device float*weights [[buffer(2)]],constant uint*p [[buffer(3)]],uint lane [[thread_index_in_simdgroup]]) {
 float local=-INFINITY;uint index=0;
 for(uint e=lane;e<p[0];e+=32)if(l[e]>local){local=l[e];index=e;}
 float sum=0;
 for(uint k=0;k<p[1];++k){
  float score=simd_max(local);uint best=simd_min(local==score?index:0xffffffffu);
  if(!lane){ids[k]=best;weights[k]=exp(score-l[ids[0]]);sum+=weights[k];}
  if(lane==best%32){
   float prevscore=score;uint previdx=best;local=-INFINITY;index=0;
   for(uint e=lane;e<p[0];e+=32){float v=l[e];bool eligible=(v<prevscore)||(v==prevscore&&e>previdx);if(eligible&&v>local){local=v;index=e;}}
  }
 }
 if(!lane)for(uint k=0;k<p[1];++k)weights[k]/=sum;
}

kernel void project_experts(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device const uint*ids [[buffer(2)]],device float*y [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]){
 uint output_row=g.x*4+simd;if(output_row>=p[1])return;
 device const uchar*row=w+(ulong(ids[g.y])*p[1]+output_row)*p[2];float sum=dot_quant_row(row,x+g.y*p[4],p[0],lane);sum=simd_sum(sum);if(!lane)y[g.y*p[1]+output_row]=sum;
}
kernel void mul_gate(device const float*x [[buffer(0)]],device const float*g [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint i [[thread_position_in_grid]]){if(i<p[0])y[i]=x[i]*sigmoid_f(g[i]);}
kernel void convolution(device const float*x [[buffer(0)]],device const uchar*w [[buffer(1)]],device float*history [[buffer(2)]],device float*y [[buffer(3)]],constant uint*p [[buffer(4)]],uint c [[thread_position_in_grid]]){
 if(c>=p[0])return;uint taps=p[1],dil=p[2],hist=(taps-1)*dil;float v=x[c]*weight_at(w+ulong(c)*p[4],p[3],taps-1);for(uint k=0;k<taps-1;++k)v+=weight_at(w+ulong(c)*p[4],p[3],k)*history[c*hist+k*dil];for(uint j=0;j+1<hist;++j)history[c*hist+j]=history[c*hist+j+1];if(hist)history[c*hist+hist-1]=x[c];y[c]=v*sigmoid_f(v);
}
kernel void delta_net(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device const float*a [[buffer(3)]],device const float*beta [[buffer(4)]],device const float*dt [[buffer(5)]],device const float*A [[buffer(6)]],device float*state [[buffer(7)]],device float*y [[buffer(8)]],constant uint*p [[buffer(9)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
 uint d=p[0],h=g.y,j=g.x,kh=h%p[1];float z=a[h]+dt[h];float soft=max(z,0.0f)+log(1.0f+exp(-abs(z)));float decay=exp(A[h]*soft);device float*s=state+(ulong(h)*d+j)*d;
 float pred=0;for(uint i=lane;i<d;i+=32){s[i]*=decay;pred+=s[i]*k[kh*d+i];}pred=simd_sum(pred);float delta=(v[h*d+j]-pred)*sigmoid_f(beta[h]);float out=0;for(uint i=lane;i<d;i+=32){s[i]+=k[kh*d+i]*delta;out+=s[i]*q[kh*d+i];}out=simd_sum(out);if(!lane)y[h*d+j]=out*rsqrt(float(d));
}
kernel void ple_gate(device const float*k [[buffer(0)]],device const float*q [[buffer(1)]],device const float*v [[buffer(2)]],device float*y [[buffer(3)]],constant uint*p [[buffer(4)]],uint h [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){uint n=p[0];float s=0;for(uint i=lane;i<n;i+=32)s+=k[h*n+i]*q[h*n+i];s=simd_sum(s)*rsqrt(float(n));float a=s==0?0:copysign(sqrt(max(abs(s),1e-6f)),s);float gate=sigmoid_f(a);for(uint i=lane;i<n;i+=32)y[h*n+i]=v[i]*gate;}
kernel void add_three(device const float*a [[buffer(0)]],device const float*b [[buffer(1)]],device const float*c [[buffer(2)]],device float*y [[buffer(3)]],constant uint*p [[buffer(4)]],uint i [[thread_position_in_grid]]){if(i<p[0])y[i]=a[i]+b[i]+c[i];}
// Text IMRoPE uses the same position on all four axes; split-half partial rotation.
kernel void rope(device const float*x [[buffer(0)]],device float*y [[buffer(1)]],constant uint*p [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i>=p[0]*p[1])return;uint j=i%p[0],r=p[2],base=i-j;if(j>=r){y[i]=x[i];return;}uint f=j%(r/2);float theta=float(p[3])*pow(as_type<float>(p[4]),-2.0f*float(f)/float(r));float c=cos(theta),s=sin(theta);y[i]=j<r/2?x[base+f]*c-x[base+f+r/2]*s:x[base+f]*s+x[base+f+r/2]*c;}
kernel void extract_gate(device const float*x [[buffer(0)]],device float*y [[buffer(1)]],constant uint*p [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<p[0]*p[1])y[i]=x[(i/p[0])*p[0]*2+p[0]+i%p[0]];}
kernel void cache_write(device const float*x [[buffer(0)]],device float*y [[buffer(1)]],constant uint*p [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<p[0])y[ulong(p[1])*p[0]+i]=x[i];}
kernel void pool_index(device const float*cache [[buffer(0)]],device float*out [[buffer(1)]],constant uint*p [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i>=p[0])return;float v=0;for(uint t=0;t<p[1];++t)v+=cache[(ulong(p[2])*p[1]+t)*p[0]+i];out[i]=v/p[1];}
kernel void index_score(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device float*score [[buffer(2)]],constant uint*p [[buffer(3)]],uint b [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){float total=0;for(uint h=0;h<p[1];++h){float v=0;for(uint j=lane;j<p[0];j+=32)v+=q[h*p[0]+j]*k[b*p[0]+j];v=simd_sum(v);total+=max(v,0.0f);}if(!lane)score[b]=total;}
kernel void index_mask(device const float*score [[buffer(0)]],device float*mask [[buffer(1)]],constant uint*p [[buffer(2)]],uint b [[thread_position_in_grid]]){
 uint length=p[0],r=p[1],blocks=length/r,tail=length%r; if(b>=blocks)return;uint rank=0;for(uint j=0;j<blocks;++j)rank+=score[j]>score[b]||(score[j]==score[b]&&j<b);uint budget=p[2]+r-1;for(uint j=0;j<r;++j)mask[b*r+j]=rank*r+j+tail<budget?1.0f:0.0f;if(!b)for(uint j=blocks*r;j<length;++j)mask[j]=1.0f;
}
kernel void attention_scores(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*mask [[buffer(2)]],device float*s [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
 uint d=p[0],h=g.y,t=g.x,kh=h/(p[1]/p[2]);if(mask[t]<=0){if(!lane)s[h*p[3]+t]=-INFINITY;return;}float v=0;for(uint j=lane;j<d;j+=32)v+=q[h*d+j]*k[(ulong(t)*p[2]+kh)*d+j];v=simd_sum(v);if(!lane)s[h*p[3]+t]=mask[t]>0?v*rsqrt(float(d)):-INFINITY;
}
kernel void softmax_rows(device const float*x [[buffer(0)]],device float*y [[buffer(1)]],constant uint*p [[buffer(2)]],uint h [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){uint n=p[0];float m=-INFINITY;for(uint i=lane;i<n;i+=32)m=max(m,x[h*n+i]);m=simd_max(m);float s=0;for(uint i=lane;i<n;i+=32)s+=exp(x[h*n+i]-m);s=simd_sum(s);for(uint i=lane;i<n;i+=32)y[h*n+i]=exp(x[h*n+i]-m)/s;}
kernel void attention_values(device const float*s [[buffer(0)]],device const float*v [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){uint d=p[0],h=g.y,j=g.x,kh=h/(p[1]/p[2]);float z=0;for(uint t=lane;t<p[3];t+=32){float probability=s[h*p[3]+t];if(probability!=0)z+=probability*v[(ulong(t)*p[2]+kh)*d+j];}z=simd_sum(z);if(!lane)y[h*d+j]=z;}
kernel void copy_range(device const float*x [[buffer(0)]],device float*y [[buffer(1)]],constant uint*p [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<p[0])y[i]=x[p[1]+i];}
kernel void embedding(device const uchar*w [[buffer(0)]],device float*y [[buffer(1)]],constant uint*p [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<p[0])y[i]=weight_at(w+ulong(p[4])*p[2],p[3],i);}

kernel void copy_into(device const float*x [[buffer(0)]],device float*y [[buffer(1)]],constant uint*p [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<p[0])y[p[1]+i]=x[i];}

// One dispatch for every [token, selected expert] pair. IDs never leave the GPU.
kernel void project_routed_batch(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device const uint*ids [[buffer(2)]],device float*y [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
 if(projection_type==16 || projection_type==22 || projection_type==42) {
  uint row=group.x*16+simd*4;if(row>=p[1])return;
  uint pair=group.y, input=p[5]?pair:pair/p[4];
  device const uchar*weight=w+(ulong(ids[pair])*p[1]+row)*p[2];
  float4 value;
  if(projection_type==42)value=dot_q2_rows(weight,x+ulong(input)*p[0],p[0],p[2],min(4u,p[1]-row),lane);
  else value=dot_iq2_rows(weight,x+ulong(input)*p[0],p[0],p[2],min(4u,p[1]-row),lane);
  value.x=simd_sum(value.x);value.y=simd_sum(value.y);value.z=simd_sum(value.z);value.w=simd_sum(value.w);
  if(!lane)for(uint r=0;r<min(4u,p[1]-row);++r)y[ulong(pair)*p[1]+row+r]=value[r];
  return;
 }

 uint row=group.x*4+simd;if(row>=p[1])return;
 uint pair=group.y, token=pair/p[4], input=p[5]?pair:token;
 device const uchar*weight=w+(ulong(ids[pair])*p[1]+row)*p[2];
 float value=simd_sum(dot_quant_row(weight,x+ulong(input)*p[0],p[0],lane));
 if(!lane)y[ulong(pair)*p[1]+row]=value;
}

// Experimental F32 SIMD-group matrix path; no F16 conversion of activations/weights.
kernel void quant_gemm(device const uchar*w [[buffer(0)]], device const float*x [[buffer(1)]], device float*y [[buffer(2)]], constant uint*p [[buffer(3)]], uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
 threadgroup float wa[16*32], xb[16*32], yc[16*16];
 uint row0=group.x*16,token0=group.y*16;
 simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8>(0.0f);
 for(uint base=0;base<p[0];base+=32) {
  for(uint i=tid;i<16*32;i+=128) {
   uint row=row0+i/32,k=base+i%32,token=token0+i/32;
   wa[i]=(row<p[1]&&k<p[0])?weight_at(w+(ulong(p[4])*p[1]+row)*p[2],projection_type,k):0.0f;
   xb[i]=(token<p[5]&&k<p[0])?x[ulong(token)*p[0]+k]:0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint k=0;k<32;k+=8) {
   simdgroup_float8x8 a,b;
   simdgroup_load(a,wa+(simd/2)*8*32+k,32,0,false);
   simdgroup_load(b,xb+(simd%2)*8*32+k,32,0,true);
   simdgroup_multiply_accumulate(acc,a,b,acc);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 simdgroup_store(acc,yc+(simd/2)*8*16+(simd%2)*8,16,0,false);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint i=tid;i<16*16;i+=128) {uint row=row0+i/16,token=token0+i%16;if(row<p[1]&&token<p[5])y[ulong(token)*p[1]+row]=yc[i];}
}

// Packed prefill operators; [token, channel] layouts, sequential recurrence.
kernel void embedding_batch(device const uchar*w [[buffer(0)]], device const uint*ids [[buffer(1)]], device float*y [[buffer(2)]], constant uint*p [[buffer(3)]], uint i [[thread_position_in_grid]]) {if(i<p[0]*p[3])y[i]=weight_at(w+ulong(ids[i/p[0]])*p[1],p[2],i%p[0]);}
kernel void repeat_emb_batch(device const float*x [[buffer(0)]],device float*y [[buffer(1)]],constant uint*p [[buffer(2)]],uint i [[thread_position_in_grid]]) {if(i<p[0]*p[1]*p[2])y[i]=x[(i/(p[0]*p[1]))*p[0]+i%p[0]];}
kernel void hc_collapse_batch(device const float*x [[buffer(0)]],device const float*g [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint i [[thread_position_in_grid]]){if(i>=p[0]*p[2])return;uint t=i/p[0],j=i%p[0];float v=0;for(uint h=0;h<p[1];++h){uint k=(t*p[1]+h)*p[0]+j;v+=x[k]*g[k];}y[i]=v/p[1];}
kernel void hc_add_batch(device const float*x [[buffer(0)]],device const float*b [[buffer(1)]],device const float*g [[buffer(2)]],device float*y [[buffer(3)]],constant uint*p [[buffer(4)]],uint i [[thread_position_in_grid]]){if(i<p[0]*p[1]*p[2]){uint t=i/(p[0]*p[1]);y[i]=x[i]+b[t*p[0]+i%p[0]]*2.0f*sigmoid_f(g[i/p[0]]/p[1]);}}
kernel void convolution_batch(device const float*x [[buffer(0)]],device const uchar*w [[buffer(1)]],device float*history [[buffer(2)]],device float*y [[buffer(3)]],device float*snap [[buffer(4)]],constant uint*p [[buffer(5)]],uint c [[thread_position_in_grid]]){
 if(c>=p[0])return;uint taps=p[1],dil=p[2],hist=(taps-1)*dil;
 for(uint t=0;t<p[5];++t){float input=x[t*p[0]+c],v=input*weight_at(w+ulong(c)*p[4],p[3],taps-1);for(uint k=0;k<taps-1;++k)v+=weight_at(w+ulong(c)*p[4],p[3],k)*history[c*hist+k*dil];for(uint j=0;j+1<hist;++j)history[c*hist+j]=history[c*hist+j+1];if(hist)history[c*hist+hist-1]=input;y[t*p[0]+c]=v*sigmoid_f(v);if(t<p[6])for(uint j=0;j<hist;++j)snap[(ulong(t)*p[0]+c)*hist+j]=history[c*hist+j];}
}
kernel void rms_tokens(device const float*x [[buffer(0)]],device const float*w [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
 uint n=p[0],t=row/p[5],h=row%p[5],offset=t*p[4]+p[2]+h*p[3];float sum=0;for(uint j=lane;j<n;j+=32){float v=x[offset+j];sum+=v*v;}sum=simd_sum(sum);float inv=rsqrt(sum/(p[7]?1.0f:float(n))+as_type<float>(p[6]));for(uint j=lane;j<n;j+=32)y[row*n+j]=x[offset+j]*inv*(p[7]?1.0f:w[(h*n+j)%p[1]]);
}
// The Qwen shape is validated as d=128 at model loading. Four values per lane
// stay in F32 registers across tokens; recurrent update and SIMD sum order match.
kernel void delta_gates_batch(device const float*a [[buffer(0)]],device const float*beta [[buffer(1)]],device const float*dt [[buffer(2)]],device const float*A [[buffer(3)]],device float*gates [[buffer(4)]],constant uint*p [[buffer(5)]],uint i [[thread_position_in_grid]]) {
 if(i>=p[0]*p[1])return;uint h=i%p[0];float z=a[i]+dt[h],soft=max(z,0.0f)+log(1.0f+exp(-abs(z)));
 gates[i*2]=exp(A[h]*soft);gates[i*2+1]=sigmoid_f(beta[i]);
}
// Each SIMD group updates four state rows of one head so k/q loads are shared;
// every row keeps the original per-token order.
kernel void delta_net_batch(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device const float*gates [[buffer(3)]],device float*state [[buffer(4)]],device float*y [[buffer(5)]],device float*snap [[buffer(6)]],constant uint*p [[buffer(7)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 constexpr uint R=4;
 uint d=p[0],nk=p[1],nv=p[2],h=g.y,j0=(g.x*4+sg)*R,kh=h%nk;if(j0>=d)return;
 float values[R][4];
 for(uint row=0;row<R;++row)for(uint r=0;r<4;++r)values[row][r]=state[(ulong(h)*d+j0+row)*d+lane+r*32];
 for(uint t=0;t<p[3];++t){float decay=gates[(t*nv+h)*2],beta=gates[(t*nv+h)*2+1];uint qi=(t*nk+kh)*d;
  float kt[4],qt[4];for(uint r=0;r<4;++r){kt[r]=k[qi+lane+r*32];qt[r]=q[qi+lane+r*32];}
  #pragma unroll
  for(uint row=0;row<R;++row){uint j=j0+row;
   float pred=0;for(uint r=0;r<4;++r){values[row][r]*=decay;pred+=values[row][r]*kt[r];}pred=simd_sum(pred);
   float delta=(v[t*p[4]+2*nk*d+h*d+j]-pred)*beta;float out=0;
   for(uint r=0;r<4;++r){uint i=lane+r*32;values[row][r]+=kt[r]*delta;out+=values[row][r]*qt[r];if(t<p[5])snap[((ulong(t)*nv+h)*d+j)*d+i]=values[row][r];}
   out=simd_sum(out);if(!lane)y[(t*nv+h)*d+j]=out*rsqrt(float(d));
  }
 }
 for(uint row=0;row<R;++row)for(uint r=0;r<4;++r)state[(ulong(h)*d+j0+row)*d+lane+r*32]=values[row][r];
}

kernel void ple_gate_batch(device const float*k [[buffer(0)]],device const float*q [[buffer(1)]],device const float*v [[buffer(2)]],device float*y [[buffer(3)]],constant uint*p [[buffer(4)]],uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {uint n=p[0],t=row/p[1];float s=0;for(uint i=lane;i<n;i+=32)s+=k[row*n+i]*q[row*n+i];s=simd_sum(s)*rsqrt(float(n));float a=s==0?0:copysign(sqrt(max(abs(s),1e-6f)),s),gate=sigmoid_f(a);for(uint i=lane;i<n;i+=32)y[row*n+i]=v[t*n+i]*gate;}
kernel void route_gpu_batch(device const float*logits [[buffer(0)]],device uint*all_ids [[buffer(1)]],device float*all_weights [[buffer(2)]],constant uint*p [[buffer(3)]],uint t [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
 device const float*l=logits+t*p[0];device uint*ids=all_ids+t*p[1];device float*weights=all_weights+t*p[1];float sum=0;
 for(uint k=0;k<p[1];++k){float local=-INFINITY;uint index=0;for(uint e=lane;e<p[0];e+=32){bool used=false;for(uint j=0;j<k;++j)used|=ids[j]==e;if(!used&&l[e]>local){local=l[e];index=e;}}float score=simd_max(local);uint best=simd_min(local==score?index:0xffffffffu);if(!lane){ids[k]=best;weights[k]=exp(score-l[ids[0]]);sum+=weights[k];}simdgroup_barrier(mem_flags::mem_device);}if(!lane)for(uint k=0;k<p[1];++k)weights[k]/=sum;
}
kernel void gather_rows(device const float*x [[buffer(0)]],device const uint*ids [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint i [[thread_position_in_grid]]){if(i<p[0]*p[1])y[i]=x[ulong(ids[i/p[0]]/p[2])*p[0]+i%p[0]];}
kernel void scatter_rows(device const float*x [[buffer(0)]],device const uint*ids [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint i [[thread_position_in_grid]]){if(i<p[0]*p[1])y[ulong(ids[i/p[0]])*p[0]+i%p[0]]=x[i];}
kernel void combine_batch(device const float*routed [[buffer(0)]],device const float*shared [[buffer(1)]],device const float*weights [[buffer(2)]],device const float*gate [[buffer(3)]],device float*out [[buffer(4)]],constant uint*p [[buffer(5)]],uint i [[thread_position_in_grid]]){if(i>=p[0]*p[2])return;uint t=i/p[0],j=i%p[0];float sum=0;for(uint e=0;e<p[1];++e)sum+=routed[(t*p[1]+e)*p[0]+j]*weights[t*p[1]+e];out[i]=sum+shared[i]/(1.0f+exp(-gate[t]));}

// 64 output rows x NT tokens. This packed 8x8 layout and register tiling
// are adapted from ggml kernel_mul_mm (fixed revision in NOTICE). This candidate
// rounds operand tiles to F16 and retains F32 accumulation.
template<uint NT>
inline void tiled_projection(device const uchar*w,device const float*x,device float*y,
                             uint cols,uint rows,uint stride,uint expert,uint row0,
                             uint offset,uint count,uint tid,uint simd,
                             threadgroup half*sa,threadgroup half*sb,threadgroup float*tail) {
    constexpr uint MB=NT/16;
    uint nr=min(64u,rows-row0),nt=min(NT,count);
    uint lr=min(tid/2,nr-1),half_k=tid%2;
    simdgroup_half8x8 ma[4],mb[MB]; simdgroup_float8x8 acc[4*MB];
    for(uint i=0;i<4*MB;++i)acc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);
    for(uint base=0;base<cols;base+=32) {
        float4x4 decoded=weights16(w+(ulong(expert)*rows+row0+lr)*stride,base+16*half_k,cols);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for(uint i=0;i<16;++i) {
            uint sx=2*half_k+i/8,sy=(tid/2)/8,lx=(tid/2)%8,ly=i%8;
            *(sa+64*(8*sx+sy)+8*ly+lx)=decoded[i/4][i%4];
        }
        if(tid<NT*4) {
            uint token=tid/4,k0=8*(tid%4),ib=NT/8*(tid%4)+token/8;
            if(token<nt && base+k0+8<=cols && cols%4==0) {
                *reinterpret_cast<threadgroup half2x4*>(sb+64*ib+8*(token%8))=
                    half2x4(*reinterpret_cast<device const float2x4*>(x+ulong(offset+token)*cols+base+k0));
            } else {
                #pragma unroll
                for(uint i=0;i<8;++i)*(sb+64*ib+8*(token%8)+i)=(token<nt&&base+k0+i<cols)?x[ulong(offset+token)*cols+base+k0+i]:0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half*a=sa+256*(simd%2);
        threadgroup const half*b=sb+64*MB*(simd/2);
        #pragma unroll
        for(uint k=0;k<4;++k) {
            #pragma unroll
            for(uint i=0;i<4;++i)simdgroup_load(ma[i],a+64*i,8,0,false);
            #pragma unroll
            for(uint i=0;i<MB;++i)simdgroup_load(mb[i],b+64*i,8,0,false);
            #pragma unroll
            for(uint i=0;i<4*MB;++i)simdgroup_multiply_accumulate(acc[i],mb[i/4],ma[i%4],acc[i]);
            a+=512;b+=NT*8;
        }
    }
    if(nr==64&&nt==NT) {
        device float*dest=y+ulong(offset+8*MB*(simd/2))*rows+row0+32*(simd%2);
        for(uint i=0;i<4*MB;++i)simdgroup_store(acc[i],dest+8*(i%4)+ulong(8*(i/4))*rows,rows,0,false);
    } else {
        // The tail aliases operand storage: all SIMD groups must finish loading
        // their final operands before any group starts writing the output tile.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float*dest=tail+32*(simd%2)+8*MB*(simd/2)*64;
        for(uint i=0;i<4*MB;++i)simdgroup_store(acc[i],dest+8*(i%4)+8*64*(i/4),64,0,false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=tid;i<nt*nr;i+=128)y[ulong(offset+i/nr)*rows+row0+i%nr]=tail[(i/nr)*64+i%nr];
    }
}

// Metadata: [expert, first packed row, rows in tile, matrix mode].
kernel void quant_expert_gemm(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device const uint*tiles [[buffer(2)]],device float*y [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    uint expert=tiles[group.y*4],offset=tiles[group.y*4+1],count=tiles[group.y*4+2],row0=group.x*64;
    if(tiles[group.y*4+3]==0) {
        for(uint r=simd;r<64;r+=4) {
            uint row=row0+r;if(row>=p[1])continue;
            device const uchar*wr=w+(ulong(expert)*p[1]+row)*p[2];
            for(uint t=0;t<count;++t) {
                float value=simd_sum(dot_quant_row(wr,x+ulong(offset+t)*p[0],p[0],lane));
                if(!lane)y[ulong(offset+t)*p[1]+row]=value;
            }
        }
        return;
    }
    threadgroup float scratch[64*32];
    if(count>16) tiled_projection<32>(w,x,y,p[0],p[1],p[2],expert,row0,offset,count,tid,simd,reinterpret_cast<threadgroup half*>(scratch),reinterpret_cast<threadgroup half*>(scratch)+64*32,scratch);
    else tiled_projection<16>(w,x,y,p[0],p[1],p[2],expert,row0,offset,count,tid,simd,reinterpret_cast<threadgroup half*>(scratch),reinterpret_cast<threadgroup half*>(scratch)+64*32,scratch);
}

// Packed causal QSA. Sparse storage preserves the original t % 32 reduction lanes.
kernel void rope_batch(device const float*x [[buffer(0)]],device float*y [[buffer(1)]],constant uint*p [[buffer(2)]],uint i [[thread_position_in_grid]]){
 uint width=p[0]*p[1];if(i>=width*p[5])return;uint j=i%p[0],r=p[2],base=i-j,pos=p[3]+(i/width)*p[6];if(j>=r){y[i]=x[i];return;}uint f=j%(r/2);float theta=float(pos)*pow(as_type<float>(p[4]),-2.0f*float(f)/float(r));float c=cos(theta),s=sin(theta);y[i]=j<r/2?x[base+f]*c-x[base+f+r/2]*s:x[base+f]*s+x[base+f+r/2]*c;
}
kernel void extract_gate_batch(device const float*x [[buffer(0)]],device float*y [[buffer(1)]],constant uint*p [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<p[0]*p[1]*p[2])y[i]=x[(i/p[0])*p[0]*2+p[0]+i%p[0]];}
kernel void pool_index_batch(device const float*cache [[buffer(0)]],device float*out [[buffer(1)]],constant uint*p [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i>=p[0]*p[3])return;uint b=p[2]+i/p[0],j=i%p[0];float v=0;for(uint t=0;t<p[1];++t)v+=cache[(ulong(b)*p[1]+t)*p[0]+j];out[i]=v/p[1];}
kernel void index_score_batch(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device float*score [[buffer(2)]],constant uint*p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
 uint b=g.x,t=g.y,blocks=(p[3]+t+1)/p[4];if(b>=blocks)return;float total=0;for(uint h=0;h<p[1];++h){float v=0;for(uint j=lane;j<p[0];j+=32)v+=q[(t*p[1]+h)*p[0]+j]*k[b*p[0]+j];v=simd_sum(v);total+=max(v,0.0f);}if(!lane)score[t*p[2]+b]=total;
}
kernel void index_mask_batch(device const float*score [[buffer(0)]],device float*mask [[buffer(1)]],constant uint*p [[buffer(2)]],uint i [[thread_position_in_grid]]){
 if(i>=p[4]*p[5])return;uint t=i/p[5],b=i%p[5],length=p[3]+t+1,r=p[1],blocks=length/r,tail=length%r;
 if(!b)for(uint j=blocks*r;j<length;++j)mask[t*p[0]+j]=1.0f;if(b>=blocks)return;
 uint rank=0;for(uint j=0;j<blocks;++j)rank+=score[t*p[5]+j]>score[t*p[5]+b]||(score[t*p[5]+j]==score[t*p[5]+b]&&j<b);
 uint budget=p[2]+r-1;for(uint j=0;j<r;++j)mask[t*p[0]+b*r+j]=rank*r+j+tail<budget?1.0f:0.0f;
}
// Stable descending score, ascending block ID. One bounded bitonic sort per query.
// At most 4096 pooled blocks (16K positions for the current ratio=4).
kernel void index_mask_sorted(device const float*score [[buffer(0)]],device float*mask [[buffer(1)]],constant uint*p [[buffer(2)]],uint t [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 threadgroup float values[4096];threadgroup uint ids[4096];
 uint length=p[3]+t+1,r=p[1],blocks=length/r,tail=length%r,n=1;while(n<blocks)n*=2;
 for(uint i=tid;i<n;i+=256){values[i]=i<blocks?score[t*p[5]+i]:-INFINITY;ids[i]=i<blocks?i:0xffffffffu;}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint size=2;size<=n;size*=2){for(uint stride=size/2;stride;stride/=2){
  for(uint i=tid;i<n;i+=256){uint j=i^stride;if(j>i){float a=values[i],b=values[j];uint ai=ids[i],bi=ids[j];bool before=a>b||(a==b&&ai<bi);bool ascending=(i&size)!=0;if(before==ascending){values[i]=b;values[j]=a;ids[i]=bi;ids[j]=ai;}}}
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }}
 if(!tid)for(uint j=blocks*r;j<length;++j)mask[t*p[0]+j]=1.0f;
 uint budget=p[2]+r-1;
 for(uint rank=tid;rank<blocks;rank+=256){uint b=ids[rank];for(uint j=0;j<r;++j)mask[t*p[0]+b*r+j]=rank*r+j+tail<budget?1.0f:0.0f;}
}

kernel void compact_mask(device const float*mask [[buffer(0)]],device uint*indices [[buffer(1)]],device uint*offsets [[buffer(2)]],constant uint*p [[buffer(3)]],uint t [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
 uint length=p[1]+t+1,count=0;for(uint j=lane;j<length;j+=32)count+=mask[t*p[0]+j]>0;
 threadgroup uint counts[32];counts[lane]=count;simdgroup_barrier(mem_flags::mem_threadgroup);uint offset=0;for(uint j=0;j<lane;++j)offset+=counts[j];offsets[t*33+lane]=offset;if(lane==31)offsets[t*33+32]=offset+count;
 for(uint j=lane;j<length;j+=32)if(mask[t*p[0]+j]>0)indices[t*p[2]+offset++]=j;
}
kernel void attention_scores_compact(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const uint*indices [[buffer(2)]],device const uint*offsets [[buffer(3)]],device float*s [[buffer(4)]],constant uint*p [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
 uint d=p[0],t=g.y/p[1],h=g.y%p[1],kh=h/(p[1]/p[2]),j=g.x;if(j>=offsets[t*33+32])return;uint pos=indices[t*p[3]+j];float v=0;for(uint i=lane;i<d;i+=32)v+=q[g.y*d+i]*k[(ulong(pos)*p[2]+kh)*d+i];v=simd_sum(v);if(!lane)s[g.y*p[3]+j]=v*rsqrt(float(d));
}
kernel void softmax_compact(device const float*x [[buffer(0)]],device const uint*offsets [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
 uint t=row/p[1],start=offsets[t*33+lane],end=offsets[t*33+lane+1];float m=-INFINITY;for(uint j=start;j<end;++j)m=max(m,x[row*p[0]+j]);m=simd_max(m);float s=0;for(uint j=start;j<end;++j)s+=exp(x[row*p[0]+j]-m);s=simd_sum(s);for(uint j=start;j<end;++j)y[row*p[0]+j]=exp(x[row*p[0]+j]-m)/s;
}
kernel void attention_values_compact(device const float*s [[buffer(0)]],device const float*v [[buffer(1)]],device const uint*indices [[buffer(2)]],device const uint*offsets [[buffer(3)]],device float*y [[buffer(4)]],constant uint*p [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
 uint d=p[0],t=g.y/p[1],h=g.y%p[1],j=g.x,kh=h/(p[1]/p[2]),start=offsets[t*33+lane],end=offsets[t*33+lane+1];float z=0;for(uint n=start;n<end;++n){float probability=s[g.y*p[3]+n];if(probability!=0){uint pos=indices[t*p[3]+n];z+=probability*v[(ulong(pos)*p[2]+kh)*d+j];}}z=simd_sum(z);if(!lane)y[g.y*d+j]=z;
}

// Causal dense QSA before the sparse budget becomes active.
kernel void attention_scores_dense_batch(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device float*s [[buffer(2)]],constant uint*p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
 uint d=p[0],t=g.y/p[1],h=g.y%p[1],kh=h/(p[1]/p[2]),j=g.x;if(j>=p[4]+t+1){if(!lane)s[g.y*p[3]+j]=-INFINITY;return;}float v=0;for(uint i=lane;i<d;i+=32)v+=q[g.y*d+i]*k[(ulong(j)*p[2]+kh)*d+i];v=simd_sum(v);if(!lane)s[g.y*p[3]+j]=v*rsqrt(float(d));
}
kernel void softmax_dense_batch(device const float*x [[buffer(0)]],device float*y [[buffer(1)]],constant uint*p [[buffer(2)]],uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
 uint length=p[2]+row/p[1]+1,stride=p[0];float m=-INFINITY;for(uint j=lane;j<length;j+=32)m=max(m,x[row*stride+j]);m=simd_max(m);float s=0;for(uint j=lane;j<length;j+=32)s+=exp(x[row*stride+j]-m);s=simd_sum(s);for(uint j=lane;j<length;j+=32)y[row*stride+j]=exp(x[row*stride+j]-m)/s;
}
kernel void attention_values_dense_batch(device const float*s [[buffer(0)]],device const float*v [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
 uint d=p[0],t=g.y/p[1],h=g.y%p[1],j=g.x,kh=h/(p[1]/p[2]),length=p[4]+t+1;float z=0;for(uint k=lane;k<length;k+=32){float probability=s[g.y*p[3]+k];if(probability!=0)z+=probability*v[(ulong(k)*p[2]+kh)*d+j];}z=simd_sum(z);if(!lane)y[g.y*d+j]=z;
}

// Dense prefill reuses each decoded weight across 32 tokens.
kernel void quant_gemm_wide(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[64*32];
    uint offset=group.y*32;
    tiled_projection<32>(w,x,y,p[0],p[1],p[2],p[4],group.x*64,offset,p[5]-offset,tid,simd,reinterpret_cast<threadgroup half*>(scratch),reinterpret_cast<threadgroup half*>(scratch)+64*32,scratch);
}

// Fixed fidelity variant: the same 64x32/K32 layout without half operand staging.
kernel void quant_gemm_wide_f32(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[(64+32)*32];
    threadgroup float*sa=scratch;
    threadgroup float*sb=scratch+64*32;
    constexpr uint NT=32,MB=2;
    uint row0=group.x*64,offset=group.y*NT;
    uint nr=min(64u,p[1]-row0),nt=min(NT,p[5]-offset);
    uint lr=min(tid/2,nr-1),half_k=tid%2;
    simdgroup_float8x8 ma[4],mb[MB],acc[4*MB];
    for(uint i=0;i<4*MB;++i)acc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);
    for(uint base=0;base<p[0];base+=32) {
        float4x4 decoded=weights16(w+(ulong(p[4])*p[1]+row0+lr)*p[2],base+16*half_k,p[0]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for(uint i=0;i<16;++i) {
            uint sx=2*half_k+i/8,sy=(tid/2)/8,lx=(tid/2)%8,ly=i%8;
            *(sa+64*(8*sx+sy)+8*ly+lx)=decoded[i/4][i%4];
        }
        if(tid<NT*4) {
            uint token=tid/4,k0=8*(tid%4),ib=NT/8*(tid%4)+token/8;
            if(token<nt && base+k0+8<=p[0] && p[0]%4==0) {
                *reinterpret_cast<threadgroup float2x4*>(sb+64*ib+8*(token%8))=
                    *reinterpret_cast<device const float2x4*>(x+ulong(offset+token)*p[0]+base+k0);
            } else {
                #pragma unroll
                for(uint i=0;i<8;++i)*(sb+64*ib+8*(token%8)+i)=(token<nt&&base+k0+i<p[0])?x[ulong(offset+token)*p[0]+base+k0+i]:0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const float*a=sa+256*(simd%2);
        threadgroup const float*b=sb+64*MB*(simd/2);
        #pragma unroll
        for(uint k=0;k<4;++k) {
            #pragma unroll
            for(uint i=0;i<4;++i)simdgroup_load(ma[i],a+64*i,8,0,false);
            #pragma unroll
            for(uint i=0;i<MB;++i)simdgroup_load(mb[i],b+64*i,8,0,false);
            #pragma unroll
            for(uint i=0;i<4*MB;++i)simdgroup_multiply_accumulate(acc[i],mb[i/4],ma[i%4],acc[i]);
            a+=512;b+=NT*8;
        }
    }
    if(nr==64&&nt==NT) {
        device float*dest=y+ulong(offset+8*MB*(simd/2))*p[1]+row0+32*(simd%2);
        for(uint i=0;i<4*MB;++i)simdgroup_store(acc[i],dest+8*(i%4)+ulong(8*(i/4))*p[1],p[1],0,false);
    } else {
        // Operand storage becomes tail storage only after every SIMD has finished reading.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float*dest=scratch+32*(simd%2)+8*MB*(simd/2)*64;
        for(uint i=0;i<4*MB;++i)simdgroup_store(acc[i],dest+8*(i%4)+8*64*(i/4),64,0,false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=tid;i<nt*nr;i+=128)y[ulong(offset+i/nr)*p[1]+row0+i%nr]=scratch[(i/nr)*64+i%nr];
    }
}

// Finite greedy reduction; numerical ties (including signed zero) use global ID.
kernel void greedy_select(device const float *logits [[buffer(0)]],
                          device const uint *mapping [[buffer(1)]],
                          device uint *partial [[buffer(2)]],
                          device uint *out [[buffer(3)]],
                          constant uint *p [[buffer(4)]],
                          uint2 group [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
    uint vocab=p[0], chunks=p[2], stage=p[3];
    float best=-INFINITY;
    uint id=0xffffffffu, bad=0;
    uint end=stage ? chunks : min(vocab, (group.x+1)*1024);
    uint start=stage ? 0 : group.x*1024;
    for(uint i=start+lane;i<end;i+=32) {
        uint off=(group.y*chunks+i)*3;
        float value=stage ? as_type<float>(partial[off]) : logits[group.y*vocab+i];
        uint token=stage ? partial[off+1] : (p[4] ? mapping[i] : i);
        bad |= stage ? partial[off+2] : uint(!isfinite(value));
        if(value>best || (value==best && token<id)) { best=value; id=token; }
    }
    float maximum=simd_max(best);
    uint winner=simd_min(best==maximum ? id : 0xffffffffu);
    bad=simd_or(bad);
    if(lane==0) {
        if(stage) { out[group.y*2]=winner; out[group.y*2+1]=bad; }
        else { uint off=(group.y*chunks+group.x)*3; partial[off]=as_type<uint>(maximum); partial[off+1]=winner; partial[off+2]=bad; }
    }
}

// [token, HC stream, (embedding || target hidden)] for canonical draft catch-up.
kernel void mtp_eh_input(device const float*en [[buffer(0)]],device const float*hn [[buffer(1)]],device float*out [[buffer(2)]],constant uint*p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 uint width=p[0],hc=p[1];if(i>=width*2*hc*p[2])return;
 uint row=i/(width*2),col=i%(width*2);
 out[i]=col<width ? en[(row/hc)*width+col] : hn[row*width+col-width];
}

// Experimental split online softmax/value accumulation. Fixed Qwen head width 256.
// p = head, heads, kv_heads, capacity, position, compact, partitions.
kernel void attention_online_parts(device const float*q [[buffer(0)]],
 device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],
 device const uint*indices [[buffer(3)]],device const uint*offsets [[buffer(4)]],
 device float*partial [[buffer(5)]],constant uint*p [[buffer(6)]],
 uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
 uint row=g.y,t=row/p[1],h=row%p[1],kh=h/(p[1]/p[2]);
 uint length=p[5]?offsets[t*33+32]:p[4]+t+1;
 uint begin=g.x*64,end=min(begin+64,length),base=(row*p[6]+g.x)*258;
 float m=-INFINITY,l=0,z[8],query[8];
 #pragma unroll
 for(uint j=0;j<8;++j){z[j]=0;query[j]=q[row*256+lane+j*32];}
 for(uint n=begin;n<end;++n){
  uint pos=p[5]?indices[t*p[3]+n]:n;float s=0;
  #pragma unroll
  for(uint j=0;j<8;++j)s+=query[j]*k[(ulong(pos)*p[2]+kh)*256+lane+j*32];
  s=simd_sum(s)*0.0625f;
  float next=max(m,s),a=exp(m-next),b=exp(s-next);
  l=l*a+b;
  #pragma unroll
  for(uint j=0;j<8;++j)z[j]=z[j]*a+b*v[(ulong(pos)*p[2]+kh)*256+lane+j*32];
  m=next;
 }
 for(uint j=0;j<8;++j)partial[base+lane+j*32]=z[j];
 if(!lane){partial[base+256]=m;partial[base+257]=l;}
}
kernel void attention_online_merge(device const float*partial [[buffer(0)]],
 device float*y [[buffer(1)]],constant uint*p [[buffer(2)]],
 uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
 uint parts=p[0];float m=-INFINITY;
 for(uint i=0;i<parts;++i)m=max(m,partial[(row*parts+i)*258+256]);
 float l=0,z[8];for(uint j=0;j<8;++j)z[j]=0;
 for(uint i=0;i<parts;++i){uint base=(row*parts+i)*258;float denominator=partial[base+257];
  if(denominator==0)continue;float weight=exp(partial[base+256]-m);l+=denominator*weight;
  for(uint j=0;j<8;++j)z[j]+=partial[base+lane+j*32]*weight;
 }
 for(uint j=0;j<8;++j)y[row*256+lane+j*32]=l>0?z[j]/l:0;
}

// Canonical ascending positions make dense and sparse query arithmetic agree,
// including a verification batch that crosses the sparse-attention boundary.
kernel void compact_mask_ordered(device const float*mask [[buffer(0)]],device uint*indices [[buffer(1)]],device uint*offsets [[buffer(2)]],constant uint*p [[buffer(3)]],uint t [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
 uint length=p[1]+t+1,base=0;
 for(uint first=0;first<length;first+=32){uint pos=first+lane;uint take=pos<length&&mask[t*p[0]+pos]>0?1:0;
  uint rank=simd_prefix_exclusive_sum(take);if(take)indices[t*p[2]+base+rank]=pos;base+=simd_sum(take);
 }
 if(!lane)offsets[t*33+32]=base;
}

// Four SIMD groups cooperate on one long BF16 row. No intermediate GPU buffer.
kernel void bf16_hc_split(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    uint row=group.x,token=group.y;
    device const ushort4*weight=reinterpret_cast<device const ushort4*>(w+(ulong(p[3])*p[1]+row)*p[2]);
    device const float*input=x+ulong(token)*p[0];
    float value=0;
    for(uint j=4*lane+128*simd;j<p[0];j+=512) {
        float4 v=as_type<float4>(uint4(weight[j/4])<<16);
        value+=v.x*input[j];value+=v.y*input[j+1];value+=v.z*input[j+2];value+=v.w*input[j+3];
    }
    value=simd_sum(value);
    threadgroup float partial[4];if(!lane)partial[simd]=value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(!lane&&!simd)y[ulong(token)*p[1]+row]=((partial[0]+partial[1])+partial[2])+partial[3];
}

// Eight queries share each64-key block. F32 Q/K/V, probabilities and accumulation.
// Matrix tiling/online rescaling follow the fixed ggml FlashAttention source in NOTICE.
// Full cache tiles and bounded key/value tails share the same attention arithmetic.
// Q queries per group (R=Q/8 row tiles). Output accumulators stay in SIMD-group registers and
// are rescaled by an exact diagonal product, so every sum keeps the per-block order
// of the former threadgroup round trip while K/V are read once per 16 queries.
template<bool Tail,uint Q>
inline void attention_prefill_impl(device const float*q, device const float*k, device const float*v,
 device const float*mask, device float*out, constant uint*p, uint2 group, uint tid, uint lane, uint sg,
 threadgroup float*sq, threadgroup float*tail, threadgroup float2*scores2, threadgroup float*rowf, threadgroup float*diag) {
    constexpr uint R=Q/8;
    uint first=group.x*Q,h=group.y,kh=h/(p[1]/p[2]),stride=p[2]*256;
    threadgroup float*scores=reinterpret_cast<threadgroup float*>(scores2);
    for(uint i=tid;i<Q*256;i+=128) {
        uint t=first+i/256;
        sq[i]=t<p[5]?q[(ulong(t)*p[1]+h)*256+i%256]:0;
    }
    float maximum[Q/4],denominator[Q/4];
    for(uint j=0;j<Q/4;++j){maximum[j]=-INFINITY;denominator[j]=0;}
    simdgroup_float8x8 accum[R][8];
    #pragma unroll
    for(uint r=0;r<R;++r)for(uint j=0;j<8;++j)accum[r][j]=make_filled_simdgroup_matrix<float,8>(0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint end=min(p[3],p[4]+min(first+Q,p[5]));
    for(uint base=0;base<end;base+=64) {
        if(Tail && base+64>p[3]) {
            for(uint i=tid;i<8*256;i+=128) {
                uint pos=(p[3]/8)*8+i/256;
                tail[i]=pos<p[3]?k[ulong(pos)*stride+kh*256+i%256]:0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        #pragma unroll
        for(uint c=0;c<2;++c) {
            uint column=8*(sg+4*c);
            simdgroup_float8x8 total[R];
            for(uint r=0;r<R;++r)total[r]=make_filled_simdgroup_matrix<float,8>(0.0f);
            for(uint j=0;j<256;j+=8) {
                simdgroup_float8x8 mq,mk=make_filled_simdgroup_matrix<float,8>(0.0f);
                if(!Tail || base+column+8<=p[3]) simdgroup_load(mk,k+ulong(base+column)*stride+kh*256+j,stride,0,true);
                else if(base+column<p[3]) simdgroup_load(mk,tail+j,256,0,true);
                #pragma unroll
                for(uint r=0;r<R;++r){simdgroup_load(mq,sq+r*8*256+j,256,0,false);simdgroup_multiply_accumulate(total[r],mq,mk,total[r]);}
            }
            #pragma unroll
            for(uint r=0;r<R;++r)simdgroup_store(total[r],scores+r*8*64+column,64,0,false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for(uint j=0;j<Q/4;++j) {
            uint qr=sg+4*j,t=first+qr;
            float2 prob=0;float factor=1;
            if(t<p[5]) {
                float2 score=scores2[qr*32+lane]*0.0625f;
                uint pos=base+2*lane;
                if(pos>p[4]+t || (p[6] && mask[ulong(t)*p[3]+pos]<=0))score.x=-INFINITY;
                if(pos+1>p[4]+t || (p[6] && mask[ulong(t)*p[3]+pos+1]<=0))score.y=-INFINITY;
                float next=max(maximum[j],simd_max(max(score.x,score.y)));
                if(next!=-INFINITY) {
                    factor=exp(maximum[j]-next);
                    prob=exp(score-next);
                    denominator[j]=denominator[j]*factor+simd_sum(prob.x+prob.y);
                    maximum[j]=next;
                }
            }
            scores2[qr*32+lane]=prob;
            if(!lane)rowf[qr]=factor;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid<64*R)diag[tid]=(tid%64)/8==tid%8?rowf[8*(tid/64)+tid%8]:0.0f;
        if(Tail && base+64>p[3]) {
            for(uint i=tid;i<8*256;i+=128) {
                uint pos=(p[3]/8)*8+i/256;
                tail[i]=pos<p[3]?v[ulong(pos)*stride+kh*256+i%256]:0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for(uint r=0;r<R;++r) {
            simdgroup_float8x8 dg;simdgroup_load(dg,diag+64*r,8,0,false);
            #pragma unroll
            for(uint j=0;j<8;++j)simdgroup_multiply(accum[r][j],dg,accum[r][j]);
        }
        for(uint c=0;c<64;c+=8) {
            simdgroup_float8x8 prob[R];
            #pragma unroll
            for(uint r=0;r<R;++r)simdgroup_load(prob[r],scores+r*8*64+c,64,0,false);
            #pragma unroll
            for(uint j=0;j<8;++j) {
                simdgroup_float8x8 value=make_filled_simdgroup_matrix<float,8>(0.0f);
                if(!Tail || base+c+8<=p[3]) simdgroup_load(value,v+ulong(base+c)*stride+kh*256+8*(sg+4*j),stride,0,false);
                else if(base+c<p[3]) simdgroup_load(value,tail+8*(sg+4*j),256,0,false);
                #pragma unroll
                for(uint r=0;r<R;++r)simdgroup_multiply_accumulate(accum[r][j],prob[r],value,accum[r][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for(uint r=0;r<R;++r)for(uint j=0;j<8;++j)simdgroup_store(accum[r][j],sq+r*8*256+8*(sg+4*j),256,0,false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for(uint j=0;j<Q/4;++j) {
        uint qr=sg+4*j,t=first+qr;
        if(t<p[5])for(uint col=lane;col<256;col+=32)
            out[(ulong(t)*p[1]+h)*256+col]=denominator[j]>0?sq[qr*256+col]/denominator[j]:0;
    }
}

kernel void attention_prefill_tiles(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device const float*mask [[buffer(3)]],device float*out [[buffer(4)]],constant uint*p [[buffer(5)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {threadgroup float sq[2048],tail[1],rowf[8],diag[64];threadgroup float2 scores2[256];attention_prefill_impl<false,8>(q,k,v,mask,out,p,group,tid,lane,sg,sq,tail,scores2,rowf,diag);}

kernel void attention_prefill_tail(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device const float*mask [[buffer(3)]],device float*out [[buffer(4)]],constant uint*p [[buffer(5)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {threadgroup float sq[2048],tail[2048],rowf[8],diag[64];threadgroup float2 scores2[256];attention_prefill_impl<true,8>(q,k,v,mask,out,p,group,tid,lane,sg,sq,tail,scores2,rowf,diag);}

kernel void quant_gemm_8(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
    uint row0=group.x*16,offset=group.y*8,nr=min(16u,p[1]-row0),nt=min(8u,p[5]-offset);
    threadgroup float sa[16*32],sb[8*32];
    simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8>(0.0f);
    for(uint base=0;base<p[0];base+=32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid<32) {
            uint row=tid/2,part=tid%2;
            float4x4 decoded=weights16(w+(ulong(p[4])*p[1]+row0+min(row,nr-1))*p[2],base+16*part,p[0]);
            #pragma unroll
            for(uint j=0;j<16;++j)sa[row*32+16*part+j]=decoded[j/4][j%4];
        } else {
            uint token=(tid-32)/4,k0=8*((tid-32)%4);
            if(token<nt && base+k0+8<=p[0] && p[0]%4==0) {
                *reinterpret_cast<threadgroup float2x4*>(sb+token*32+k0)=
                    *reinterpret_cast<device const float2x4*>(x+ulong(offset+token)*p[0]+base+k0);
            } else {
                #pragma unroll
                for(uint j=0;j<8;++j)sb[token*32+k0+j]=token<nt&&base+k0+j<p[0]?x[ulong(offset+token)*p[0]+base+k0+j]:0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for(uint j=0;j<32;j+=8) {
            simdgroup_float8x8 ma,mb;
            simdgroup_load(ma,sa+sg*8*32+j,32,0,true);
            simdgroup_load(mb,sb+j,32,0,false);
            simdgroup_multiply_accumulate(acc,mb,ma,acc);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(acc,sa+sg*8,16,0,false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=tid;j<nt*nr;j+=64)y[ulong(offset+j/nr)*p[1]+row0+j%nr]=sa[j/nr*16+j%nr];
}

// Development kernels adapted from the fixed ggml Metal implementation in NOTICE.
// Exact original quantization, F32 input and accumulation, with reordered sums.
// One weight decode serves NT tokens; each token keeps the count-one order.
template<uint NT>
inline void iq4_tokens(device const uchar*w,device const float*x,device float*y,constant uint*p,uint2 group,uint lane,uint sg,threadgroup float*lut) {
 if(sg==0 && lane<16)lut[lane]=float(kvalues_iq4nl[lane]);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 uint row0=group.x*8+sg*2;if(row0>=p[1])return;
 uint ix=lane/16,sub=(lane%16)/2,half_group=lane%2;
 float2 sums[NT];
 for(uint t=0;t<NT;++t)sums[t]=0;
 for(uint block=ix;block<p[0]/256;block+=2) {
  float4 yl[NT][4];
  #pragma unroll
  for(uint t=0;t<NT;++t) {
   device const float4*input=reinterpret_cast<device const float4*>(x+ulong(t)*p[0]+block*256+sub*32+half_group*8);
   yl[t][0]=input[0];yl[t][1]=input[4];yl[t][2]=input[1];yl[t][3]=input[5];
  }
  #pragma unroll
  for(uint row=0;row<2;++row) {
   if(row0+row>=p[1])continue;
   device const uchar*b=w+(ulong(p[4])*p[1]+row0+row)*p[2]+block*136;
   device const uint*q=reinterpret_cast<device const uint*>(b+8+sub*16+half_group*8);
   float4 low0,high0,low1,high1;
   #pragma unroll
   for(uint j=0;j<4;++j) {
    low0[j]=lut[(q[0]>>(8*j))&15];high0[j]=lut[(q[0]>>(8*j+4))&15];
    low1[j]=lut[(q[1]>>(8*j))&15];high1[j]=lut[(q[1]>>(8*j+4))&15];
   }
   uint high=uint(b[2])|(uint(b[3])<<8);
   int scale=int(((b[4+sub/2]>>(4*(sub%2)))&15)|(((high>>(2*sub))&3)<<4))-32;
   float ds=half_at(b)*scale;
   #pragma unroll
   for(uint t=0;t<NT;++t) {
    float4 acc=(yl[t][0]*low0+yl[t][2]*low1)+(yl[t][1]*high0+yl[t][3]*high1);
    sums[t][row]+=ds*(acc.x+acc.y+acc.z+acc.w);
   }
  }
 }
 #pragma unroll
 for(uint t=0;t<NT;++t) {
  float2 total=float2(simd_sum(sums[t].x),simd_sum(sums[t].y));
  if(!lane)for(uint row=0;row<min(2u,p[1]-row0);++row)y[ulong(t)*p[1]+row0+row]=total[row];
 }
}
#define IQ4_TOKENS(name,nt) kernel void name(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {threadgroup float lut[16];iq4_tokens<nt>(w,x,y,p,group,lane,sg,lut);}
IQ4_TOKENS(iq4_vector,1)
IQ4_TOKENS(iq4_vector2,2)
IQ4_TOKENS(iq4_vector3,3)
IQ4_TOKENS(iq4_vector4,4)

// IQ2_XXS tables in shared memory; same arithmetic and four-row reduction as baseline.
kernel void project_routed_lut(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device const uint*ids [[buffer(2)]],device float*y [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup ulong lut[256];threadgroup uchar sign_lut[128];
 uint tid=sg*32+lane;lut[tid]=iq2xxs_grid[tid];lut[tid+128]=iq2xxs_grid[tid+128];sign_lut[tid]=ksigns_iq2xs[tid];
 threadgroup_barrier(mem_flags::mem_threadgroup);
 uint row0=group.x*16+sg*4;if(row0>=p[1])return;
 uint pair=group.y,input=p[5]?pair:pair/p[4];
 device const uchar*rows=w+(ulong(ids[pair])*p[1]+row0)*p[2];
 device const float*data=x+ulong(input)*p[0];float4 sum=0;
 for(uint at=lane*32;at<p[0];at+=1024) {
  float4 values[8];
  #pragma unroll
  for(uint j=0;j<8;++j)values[j]=*reinterpret_cast<device const float4*>(data+at+4*j);
  #pragma unroll
  for(uint r=0;r<4;++r) {
   if(row0+r>=p[1])continue;
   uint group=(at%256)/32;device const uchar*b=rows+ulong(r)*p[2]+(at/256)*66;
   device const uchar*a=b+2+group*8;
   uint packed=uint(a[4])|(uint(a[5])<<8)|(uint(a[6])<<16)|(uint(a[7])<<24);float sub=0;
   #pragma unroll
   for(uint l=0;l<4;++l) {
    threadgroup uchar4* grid=reinterpret_cast<threadgroup uchar4*>(lut+a[l]);uint signs=sign_lut[(packed>>(7*l))&127];
    float4 lo=float4(grid[0]),hi=float4(grid[1]);
    lo=select(lo,-lo,(uint4(signs)&uint4(1,2,4,8))!=uint4(0));
    hi=select(hi,-hi,(uint4(signs)&uint4(16,32,64,128))!=uint4(0));
    float part=dot(values[2*l],lo)+dot(values[2*l+1],hi);
    sub+=(0.5f+(packed>>28))*part;
   }
   sum[r]+=half_at(b)*sub;
  }
 }
 sum*=0.25f;
 sum.x=simd_sum(sum.x);sum.y=simd_sum(sum.y);sum.z=simd_sum(sum.z);sum.w=simd_sum(sum.w);
 if(!lane)for(uint r=0;r<min(4u,p[1]-row0);++r)y[ulong(pair)*p[1]+row0+r]=sum[r];
}

kernel void project_routed_q2_packed(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device const uint*ids [[buffer(2)]],device float*y [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint row0=group.x*16+sg*4;if(row0>=p[1])return;
 uint pair=group.y,input=p[5]?pair:pair/p[4];
 device const uchar*rows=w+(ulong(ids[pair])*p[1]+row0)*p[2];
 device const float*data=x+ulong(input)*p[0];float4 sum=0;
 for(uint at=lane*16;at<p[0];at+=512) {
  float values[16];
  #pragma unroll
  for(uint j=0;j<16;++j)values[j]=data[at+j];
  #pragma unroll
  for(uint r=0;r<4;++r) {
   if(row0+r>=p[1])continue;
   device const uchar*b=rows+ulong(r)*p[2]+(at/64)*18;
   float part=0;
   #pragma unroll
   for(uint byte=0;byte<4;++byte) {
    uint packed=b[2+(at%64)/4+byte];
    #pragma unroll
    for(uint j=0;j<4;++j)part+=values[byte*4+j]*(int((packed>>(2*j))&3)-1);
   }
   sum[r]+=half_at(b)*part;
  }
 }
 sum.x=simd_sum(sum.x);sum.y=simd_sum(sum.y);sum.z=simd_sum(sum.z);sum.w=simd_sum(sum.w);
 if(!lane)for(uint r=0;r<min(4u,p[1]-row0);++r)y[ulong(pair)*p[1]+row0+r]=sum[r];
}

// Q2_0 routed rows for 640-column down projections: eight lanes own one output row
// and five 16-value chunks each, so all 32 lanes stay busy. Reduction order differs
// from project_routed_q2_packed; callers validate against F64.
kernel void routed_q2_lanes8(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device const uint*ids [[buffer(2)]],device float*y [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint row=group.x*16+sg*4+lane/8,part=lane%8;
 uint pair=group.y,input=p[5]?pair:pair/p[4];
 device const uchar*rows=w+(ulong(ids[pair])*p[1]+min(row,p[1]-1))*p[2];
 device const float*data=x+ulong(input)*p[0];float sum=0;
 for(uint chunk=part;chunk<p[0]/16;chunk+=8) {
  uint at=chunk*16;device const uchar*b=rows+(at/64)*18;
  device const ushort*q=reinterpret_cast<device const ushort*>(b+2+(at%64)/4);
  uint packed=uint(q[0])|(uint(q[1])<<16);
  device const float4*v=reinterpret_cast<device const float4*>(data+at);
  float block=0;
  #pragma unroll
  for(uint byte=0;byte<4;++byte) {
   float4 values=v[byte];uint bits=packed>>(8*byte);
   block+=values.x*(int(bits&3)-1)+values.y*(int((bits>>2)&3)-1)+values.z*(int((bits>>4)&3)-1)+values.w*(int((bits>>6)&3)-1);
  }
  sum+=half_at(b)*block;
 }
 sum+=simd_shuffle_xor(sum,4);sum+=simd_shuffle_xor(sum,2);sum+=simd_shuffle_xor(sum,1);
 if(row<p[1] && part==0)y[ulong(pair)*p[1]+row]=sum;
}

// Four IQ2_S rows with the exact arithmetic order of dot_iq2_rows (projection_type 22).
template<typename Grid>
inline void iq2s_rows_step(device const uchar*rows,thread const float4*values,uint at,uint stride,uint valid,Grid grid_table,thread float4&sum) {
 #pragma unroll
 for(uint output=0;output<4;++output) {
  if(output>=valid)continue;
  device const uchar*b=rows+ulong(output)*stride+(at/256)*82;
  uint group=(at%256)/32;float d=half_at(b),sub=0.0f;
  #pragma unroll
  for(uint l=0;l<4;++l) {
   uint index=b[2+4*group+l]|((uint(b[66+group])<<(8-2*l))&0x300);
   uint signs=b[34+4*group+l];
   float scale=0.5f+((b[74+group]>>(4*(l/2)))&15);
   ulong g=grid_table[index];
   float4 lo=float4(as_type<uchar4>(uint(g))),hi=float4(as_type<uchar4>(uint(g>>32)));
   lo=select(lo,-lo,(uint4(signs)&uint4(1,2,4,8))!=uint4(0));
   hi=select(hi,-hi,(uint4(signs)&uint4(16,32,64,128))!=uint4(0));
   float part=dot(values[2*l],lo)+dot(values[2*l+1],hi);
   sub+=scale*part;
  }
  sum[output]+=d*sub;
 }
}
inline void load_iq2s_grid(threadgroup ulong*grid,uint tid) {
 for(uint i=tid;i<1024;i+=128)grid[i]=iq2s_grid[i];
 threadgroup_barrier(mem_flags::mem_threadgroup);
}
template<typename Grid>
inline float4 iq2s_rows(device const uchar*rows,device const float*x,uint cols,uint stride,uint valid,uint lane,Grid grid) {
 float4 sum=0.0f;
 for(uint at=lane*32;at<cols;at+=32*32) {
  float4 values[8];
  #pragma unroll
  for(uint j=0;j<8;++j)values[j]=*reinterpret_cast<device const float4*>(x+at+4*j);
  iq2s_rows_step(rows,values,at,stride,valid,grid,sum);
 }
 return sum*0.25f;
}
// IQ2_S routed rows with the codebook in threadgroup memory.
kernel void routed_iq2s_tg(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device const uint*ids [[buffer(2)]],device float*y [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup ulong grid[1024];load_iq2s_grid(grid,sg*32+lane);
 uint row=group.x*16+sg*4;if(row>=p[1])return;
 uint pair=group.y,input=p[5]?pair:pair/p[4];
 float4 value=iq2s_rows(w+(ulong(ids[pair])*p[1]+row)*p[2],x+ulong(input)*p[0],p[0],p[2],min(4u,p[1]-row),lane,(threadgroup const ulong*)grid);
 value.x=simd_sum(value.x);value.y=simd_sum(value.y);value.z=simd_sum(value.z);value.w=simd_sum(value.w);
 if(!lane)for(uint r=0;r<min(4u,p[1]-row);++r)y[ulong(pair)*p[1]+row+r]=value[r];
}
// IQ2_S routed gate and up rows plus SwiGLU in one pass. Each projection keeps the
// dot_iq2_rows order and the activation matches the swiglu kernel expression.
template<bool Shared>
inline void routed_gate_up_iq2s_impl(device const uchar*gw,device const uchar*uw,device const float*x,device const uint*ids,device float*y,constant uint*p,uint2 group,uint lane,uint sg,threadgroup ulong*grid) {
 if(Shared)load_iq2s_grid(grid,sg*32+lane);
 uint row=group.x*16+sg*4;if(row>=p[1])return;
 uint pair=group.y,input=pair/p[4],valid=min(4u,p[1]-row);
 ulong offset=(ulong(ids[pair])*p[1]+row)*p[2];
 device const float*data=x+ulong(input)*p[0];
 float4 g=0.0f,u=0.0f;
 for(uint at=lane*32;at<p[0];at+=32*32) {
  float4 values[8];
  #pragma unroll
  for(uint j=0;j<8;++j)values[j]=*reinterpret_cast<device const float4*>(data+at+4*j);
  if(Shared) {
   iq2s_rows_step(gw+offset,values,at,p[2],valid,(threadgroup const ulong*)grid,g);
   iq2s_rows_step(uw+offset,values,at,p[2],valid,(threadgroup const ulong*)grid,u);
  } else {
   iq2s_rows_step(gw+offset,values,at,p[2],valid,iq2s_grid,g);
   iq2s_rows_step(uw+offset,values,at,p[2],valid,iq2s_grid,u);
  }
 }
 g*=0.25f;u*=0.25f;
 g.x=simd_sum(g.x);g.y=simd_sum(g.y);g.z=simd_sum(g.z);g.w=simd_sum(g.w);
 u.x=simd_sum(u.x);u.y=simd_sum(u.y);u.z=simd_sum(u.z);u.w=simd_sum(u.w);
 if(!lane)for(uint r=0;r<valid;++r)y[ulong(pair)*p[1]+row+r]=(g[r]/(1.0f+exp(-g[r])))*u[r];
}
kernel void routed_gate_up_iq2s(device const uchar*gw [[buffer(0)]],device const uchar*uw [[buffer(1)]],device const float*x [[buffer(2)]],device const uint*ids [[buffer(3)]],device float*y [[buffer(4)]],constant uint*p [[buffer(5)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {routed_gate_up_iq2s_impl<false>(gw,uw,x,ids,y,p,group,lane,sg,nullptr);}
kernel void routed_gate_up_iq2s_tg(device const uchar*gw [[buffer(0)]],device const uchar*uw [[buffer(1)]],device const float*x [[buffer(2)]],device const uint*ids [[buffer(3)]],device float*y [[buffer(4)]],constant uint*p [[buffer(5)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {threadgroup ulong grid[1024];routed_gate_up_iq2s_impl<true>(gw,uw,x,ids,y,p,group,lane,sg,grid);}
// IQ2_S routed rows with sixteen lanes per row pair: each lane owns 32-value groups
// at a 512-column stride, which divides the 2560-column expert input exactly.
// The reduction order differs from dot_iq2_rows; callers validate against F64.
template<bool Fused>
inline void routed_iq2s_half_impl(device const uchar*gw,device const uchar*uw,device const float*x,device const uint*ids,device float*y,constant uint*p,uint2 group,uint lane,uint sg) {
 uint row=group.x*16+sg*4+(lane/16)*2,part=lane%16;
 uint pair=group.y,input=p[5]?pair:pair/p[4];
 uint valid=row<p[1]?min(2u,p[1]-row):0;
 ulong offset=(ulong(ids[pair])*p[1]+min(row,p[1]-1))*p[2];
 device const float*data=x+ulong(input)*p[0];
 float4 g=0.0f,u=0.0f;
 for(uint at=part*32;at<p[0];at+=512) {
  float4 values[8];
  #pragma unroll
  for(uint j=0;j<8;++j)values[j]=*reinterpret_cast<device const float4*>(data+at+4*j);
  iq2s_rows_step(gw+offset,values,at,p[2],valid,iq2s_grid,g);
  if(Fused)iq2s_rows_step(uw+offset,values,at,p[2],valid,iq2s_grid,u);
 }
 float2 a=float2(g.x,g.y)*0.25f,b=float2(u.x,u.y)*0.25f;
 #pragma unroll
 for(uint mask=8;mask;mask/=2){a+=simd_shuffle_xor(a,mask);if(Fused)b+=simd_shuffle_xor(b,mask);}
 if(part==0)for(uint r=0;r<valid;++r)y[ulong(pair)*p[1]+row+r]=Fused?(a[r]/(1.0f+exp(-a[r])))*b[r]:a[r];
}
kernel void routed_iq2s_half(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device const uint*ids [[buffer(2)]],device float*y [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {routed_iq2s_half_impl<false>(w,w,x,ids,y,p,group,lane,sg);}
kernel void routed_gate_up_iq2s_half(device const uchar*gw [[buffer(0)]],device const uchar*uw [[buffer(1)]],device const float*x [[buffer(2)]],device const uint*ids [[buffer(3)]],device float*y [[buffer(4)]],constant uint*p [[buffer(5)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {routed_iq2s_half_impl<true>(gw,uw,x,ids,y,p,group,lane,sg);}

kernel void rms_wide(device const float*x [[buffer(0)]],device const float*w [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint tid=sg*32+lane,n=p[0],start=p[2]+row*p[3];float sum=0;
 for(uint j=tid;j<n;j+=128){float v=x[start+j];sum+=v*v;}sum=simd_sum(sum);
 threadgroup float partial[4];if(!lane)partial[sg]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
 if(sg==0){float total=simd_sum(lane<4?partial[lane]:0.0f);if(!lane)partial[0]=rsqrt(total/(p[5]?1.0f:float(n))+as_type<float>(p[4]));}
 threadgroup_barrier(mem_flags::mem_threadgroup);float inv=partial[0];
 for(uint j=tid;j<n;j+=128)y[row*n+j]=x[start+j]*inv*(p[5]?1.0f:w[(row*n+j)%p[1]]);
}
kernel void route_top10(device const float*logits [[buffer(0)]],device uint*ids [[buffer(1)]],device float*weights [[buffer(2)]],constant uint*p [[buffer(3)]],uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint tid=sg*32+lane;float values[4];for(uint j=0;j<4;++j)values[j]=logits[row*512+tid+128*j];
 threadgroup float best[40];threadgroup uint indices[40];
 for(uint k=0;k<10;++k){float local=-INFINITY;uint index=0xffffffffu,slot=0;for(uint j=0;j<4;++j){uint id=tid+128*j;if(values[j]>local||(values[j]==local&&id<index)){local=values[j];index=id;slot=j;}}float score=simd_max(local);uint winner=simd_min(local==score?index:0xffffffffu);if(index==winner)values[slot]=-INFINITY;if(!lane){best[sg*10+k]=score;indices[sg*10+k]=winner;}}
 threadgroup_barrier(mem_flags::mem_threadgroup);if(sg)return;
 float v0=best[lane],v1=lane<8?best[lane+32]:-INFINITY;uint id0=indices[lane],id1=lane<8?indices[lane+32]:0xffffffffu;float first=0,sum=0;
 for(uint k=0;k<10;++k){bool take0=v0>v1||(v0==v1&&id0<id1);float local=take0?v0:v1;uint index=take0?id0:id1;float score=simd_max(local);uint winner=simd_min(local==score?index:0xffffffffu);if(index==winner){if(take0)v0=-INFINITY;else v1=-INFINITY;}if(k==0)first=score;if(!lane){ids[row*10+k]=winner;float p=exp(score-first);weights[row*10+k]=p;sum+=p;}}
 if(!lane)for(uint k=0;k<10;++k)weights[row*10+k]/=sum;
}

// Packed: x rows are already in expert order (down projection); otherwise they
// are gathered from token rows. Output rows are always in packed expert order.
template<uint NT, bool Packed>
inline void indexed_projection(device const uchar*w,device const float*x,device float*y,
                             uint cols,uint rows,uint stride,uint expert,uint row0,
                             uint offset,uint count,uint tid,uint simd,device const uint*pairs,uint top,
                             threadgroup float*sa,threadgroup float*sb,threadgroup float*tail) {
    constexpr uint MB=NT/16;
    uint nr=min(64u,rows-row0),nt=min(NT,count);
    uint lr=min(tid/2,nr-1),half_k=tid%2;
    simdgroup_float8x8 ma[4],mb[MB]; simdgroup_float8x8 acc[4*MB];
    for(uint i=0;i<4*MB;++i)acc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);
    for(uint base=0;base<cols;base+=32) {
        float4x4 decoded=weights16(w+(ulong(expert)*rows+row0+lr)*stride,base+16*half_k,cols);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for(uint i=0;i<16;++i) {
            uint sx=2*half_k+i/8,sy=(tid/2)/8,lx=(tid/2)%8,ly=i%8;
            *(sa+64*(8*sx+sy)+8*ly+lx)=decoded[i/4][i%4];
        }
        if(tid<NT*4) {
            uint token=tid/4,k0=8*(tid%4),ib=NT/8*(tid%4)+token/8;
            uint source=token<nt ? (Packed ? offset+token : pairs[offset+token]/top) : 0;
            if(token<nt && base+k0+8<=cols && cols%4==0) {
                *reinterpret_cast<threadgroup float2x4*>(sb+64*ib+8*(token%8))=
                    float2x4(*reinterpret_cast<device const float2x4*>(x+ulong(source)*cols+base+k0));
            } else {
                #pragma unroll
                for(uint i=0;i<8;++i)*(sb+64*ib+8*(token%8)+i)=(token<nt&&base+k0+i<cols)?x[ulong(source)*cols+base+k0+i]:0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const float*a=sa+256*(simd%2);
        threadgroup const float*b=sb+64*MB*(simd/2);
        #pragma unroll
        for(uint k=0;k<4;++k) {
            #pragma unroll
            for(uint i=0;i<4;++i)simdgroup_load(ma[i],a+64*i,8,0,false);
            #pragma unroll
            for(uint i=0;i<MB;++i)simdgroup_load(mb[i],b+64*i,8,0,false);
            #pragma unroll
            for(uint i=0;i<4*MB;++i)simdgroup_multiply_accumulate(acc[i],mb[i/4],ma[i%4],acc[i]);
            a+=512;b+=NT*8;
        }
    }
    if(nr==64&&nt==NT) {
        device float*dest=y+ulong(offset+8*MB*(simd/2))*rows+row0+32*(simd%2);
        for(uint i=0;i<4*MB;++i)simdgroup_store(acc[i],dest+8*(i%4)+ulong(8*(i/4))*rows,rows,0,false);
    } else {
        // The tail aliases operand storage: all SIMD groups must finish loading
        // their final operands before any group starts writing the output tile.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float*dest=tail+32*(simd%2)+8*MB*(simd/2)*64;
        for(uint i=0;i<4*MB;++i)simdgroup_store(acc[i],dest+8*(i%4)+8*64*(i/4),64,0,false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint i=tid;i<nt*nr;i+=128)y[ulong(offset+i/nr)*rows+row0+i%nr]=tail[(i/nr)*64+i%nr];
    }
}


kernel void quant_expert_gather(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device const uint*tiles [[buffer(2)]],device float*y [[buffer(3)]],device const uint*pairs [[buffer(4)]],constant uint*p [[buffer(5)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
 uint expert=tiles[group.y*4],offset=tiles[group.y*4+1],count=tiles[group.y*4+2],row0=group.x*64;
 if(tiles[group.y*4+3]==0) {
  for(uint r=simd;r<64;r+=4) {
   uint row=row0+r;if(row>=p[1])continue;
   device const uchar*wr=w+(ulong(expert)*p[1]+row)*p[2];
   for(uint t=0;t<count;++t) {
    uint src=false ? offset+t : pairs[offset+t]/p[4];
    uint dst=false ? pairs[offset+t] : offset+t;
    float value=simd_sum(dot_quant_row(wr,x+ulong(src)*p[0],p[0],lane));
    if(!lane)y[ulong(dst)*p[1]+row]=value;
   }
  }
  return;
 }
 threadgroup float scratch[(64+32)*32];
 if(count>16)indexed_projection<32,false>(w,x,y,p[0],p[1],p[2],expert,row0,offset,count,tid,simd,pairs,p[4],scratch,scratch+64*32,scratch);
 else indexed_projection<16,false>(w,x,y,p[0],p[1],p[2],expert,row0,offset,count,tid,simd,pairs,p[4],scratch,scratch+64*32,scratch);
}

// Down projection with packed (expert-ordered) output rows; combine_packed
// reads them back through the inverse pair permutation.
kernel void quant_expert_packed(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device const uint*tiles [[buffer(2)]],device float*y [[buffer(3)]],device const uint*pairs [[buffer(4)]],constant uint*p [[buffer(5)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
 uint expert=tiles[group.y*4],offset=tiles[group.y*4+1],count=tiles[group.y*4+2],row0=group.x*64;
 if(tiles[group.y*4+3]==0) {
  for(uint r=simd;r<64;r+=4) {
   uint row=row0+r;if(row>=p[1])continue;
   device const uchar*wr=w+(ulong(expert)*p[1]+row)*p[2];
   for(uint t=0;t<count;++t) {
    uint src=offset+t,dst=offset+t;
    float value=simd_sum(dot_quant_row(wr,x+ulong(src)*p[0],p[0],lane));
    if(!lane)y[ulong(dst)*p[1]+row]=value;
   }
  }
  return;
 }
 threadgroup float scratch[(64+32)*32];
 if(count>16)indexed_projection<32,true>(w,x,y,p[0],p[1],p[2],expert,row0,offset,count,tid,simd,pairs,p[4],scratch,scratch+64*32,scratch);
 else indexed_projection<16,true>(w,x,y,p[0],p[1],p[2],expert,row0,offset,count,tid,simd,pairs,p[4],scratch,scratch+64*32,scratch);
}


// Gate and up of the same 64 rows share one gathered activation tile and emit
// SwiGLU directly. 256 threads: each of the eight SIMD groups owns 16 gate and
// the same 16 up rows, so its register tile matches indexed_projection's. Each
// product keeps that kernel's K order, so the result matches gather(gate),
// gather(up) and swiglu bit for bit.
template<uint NT>
inline void indexed_gate_up(device const uchar*gw,device const uchar*uw,device const float*x,device float*y,
                            uint cols,uint rows,uint stride,uint expert,uint row0,
                            uint offset,uint count,uint tid,uint simd,device const uint*pairs,uint top,
                            threadgroup float*sa,threadgroup float*sb,threadgroup float*tail) {
    constexpr uint MB=NT/16;
    uint nr=min(64u,rows-row0),nt=min(NT,count);
    uint up=tid/128,lr=min((tid%128)/2,nr-1),half_k=tid%2;
    simdgroup_float8x8 ga[2],ua[2],mb[MB],gacc[2*MB],uacc[2*MB];
    #pragma unroll
    for(uint i=0;i<2*MB;++i){gacc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);uacc[i]=gacc[i];}
    device const uchar*wrow=(up?uw:gw)+(ulong(expert)*rows+row0+lr)*stride;
    for(uint base=0;base<cols;base+=32) {
        float4x4 decoded=weights16(wrow,base+16*half_k,cols);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for(uint i=0;i<16;++i) {
            uint sx=2*half_k+i/8,sy=8*up+((tid%128)/2)/8,lx=((tid%128)/2)%8,ly=i%8;
            *(sa+64*(16*sx+sy)+8*ly+lx)=decoded[i/4][i%4];
        }
        if(tid<NT*4) {
            uint token=tid/4,k0=8*(tid%4),ib=NT/8*(tid%4)+token/8;
            uint source=token<nt ? pairs[offset+token]/top : 0;
            if(token<nt && base+k0+8<=cols && cols%4==0) {
                *reinterpret_cast<threadgroup float2x4*>(sb+64*ib+8*(token%8))=
                    float2x4(*reinterpret_cast<device const float2x4*>(x+ulong(source)*cols+base+k0));
            } else {
                #pragma unroll
                for(uint i=0;i<8;++i)*(sb+64*ib+8*(token%8)+i)=(token<nt&&base+k0+i<cols)?x[ulong(source)*cols+base+k0+i]:0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const float*a=sa+128*(simd%4);
        threadgroup const float*b=sb+64*MB*(simd/4);
        #pragma unroll
        for(uint k=0;k<4;++k) {
            #pragma unroll
            for(uint i=0;i<2;++i){simdgroup_load(ga[i],a+64*i,8,0,false);simdgroup_load(ua[i],a+512+64*i,8,0,false);}
            #pragma unroll
            for(uint i=0;i<MB;++i)simdgroup_load(mb[i],b+64*i,8,0,false);
            #pragma unroll
            for(uint i=0;i<2*MB;++i){simdgroup_multiply_accumulate(gacc[i],mb[i/2],ga[i%2],gacc[i]);simdgroup_multiply_accumulate(uacc[i],mb[i/2],ua[i%2],uacc[i]);}
            a+=1024;b+=NT*8;
        }
    }
    // SwiGLU goes through threadgroup memory: element access on the accumulators
    // (thread_elements) moves them out of registers. The staging aliases operand
    // storage, so every SIMD group must finish reading first.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float*gt=tail+16*(simd%4)+8*MB*(simd/4)*64;
    #pragma unroll
    for(uint i=0;i<2*MB;++i) {
        simdgroup_store(gacc[i],gt+8*(i%2)+8*64*(i/2),64,0,false);
        simdgroup_store(uacc[i],gt+64*NT+8*(i%2)+8*64*(i/2),64,0,false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint i=tid;i<nt*nr;i+=256) {
        float g=tail[(i/nr)*64+i%nr],u=tail[64*NT+(i/nr)*64+i%nr];
        y[ulong(offset+i/nr)*rows+row0+i%nr]=(g/(1.0f+exp(-g)))*u;
    }
}

// Fused routed gate/up/SwiGLU for packed prefill (256 threads per 64-row tile).
// Buffers: gate, up, x, tiles, y, pairs; params as quant_expert_gather.
kernel void quant_expert_gate_up(device const uchar*gw [[buffer(0)]],device const uchar*uw [[buffer(1)]],device const float*x [[buffer(2)]],device const uint*tiles [[buffer(3)]],device float*y [[buffer(4)]],device const uint*pairs [[buffer(5)]],constant uint*p [[buffer(6)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
 uint expert=tiles[group.y*4],offset=tiles[group.y*4+1],count=tiles[group.y*4+2],row0=group.x*64;
 if(tiles[group.y*4+3]==0) {
  for(uint r=simd;r<64;r+=8) {
   uint row=row0+r;if(row>=p[1])continue;
   ulong wr=(ulong(expert)*p[1]+row)*p[2];
   for(uint t=0;t<count;++t) {
    device const float*src=x+ulong(pairs[offset+t]/p[4])*p[0];
    float g=simd_sum(dot_quant_row(gw+wr,src,p[0],lane)),u=simd_sum(dot_quant_row(uw+wr,src,p[0],lane));
    if(!lane)y[ulong(offset+t)*p[1]+row]=(g/(1.0f+exp(-g)))*u;
   }
  }
  return;
 }
 threadgroup float scratch[(128+32)*32];
 if(count>16)indexed_gate_up<32>(gw,uw,x,y,p[0],p[1],p[2],expert,row0,offset,count,tid,simd,pairs,p[4],scratch,scratch+128*32,scratch);
 else indexed_gate_up<16>(gw,uw,x,y,p[0],p[1],p[2],expert,row0,offset,count,tid,simd,pairs,p[4],scratch,scratch+128*32,scratch);
}

template<uint NR>
inline void routed_vec(device const uchar*w,device const float*x,device const uint*ids,device float*y,constant uint*p,uint2 group,uint lane,uint sg) {
 uint row0=group.x*(4*NR)+sg*NR;if(row0>=p[1])return;uint pair=group.y,input=p[5]?pair:pair/p[4];float4 sum=0;
 device const float*data=x+ulong(input)*p[0];
 for(uint at=16*lane;at<p[0];at+=512) {
  device const float4*src=reinterpret_cast<device const float4*>(data+at);float4 a=src[0],b=src[1],c=src[2],d=src[3];
  #pragma unroll
  for(uint r=0;r<NR;++r) {if(row0+r>=p[1])continue;
   float4x4 v=weights16(w+(ulong(ids[pair])*p[1]+row0+r)*p[2],at,p[0]);
   sum[r]+=dot(a,v[0])+dot(b,v[1])+dot(c,v[2])+dot(d,v[3]);
  }
 }
 sum.x=simd_sum(sum.x);sum.y=simd_sum(sum.y);sum.z=simd_sum(sum.z);sum.w=simd_sum(sum.w);
 if(!lane)for(uint r=0;r<NR;++r)if(row0+r<p[1])y[ulong(pair)*p[1]+row0+r]=sum[r];
}
kernel void routed_vec4(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device const uint*ids [[buffer(2)]],device float*y [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {routed_vec<4>(w,x,ids,y,p,group,lane,sg);}

kernel void decode_blocks16(device const uchar*w [[buffer(0)]],device float*out [[buffer(1)]],constant uint*p [[buffer(2)]],uint i [[thread_position_in_grid]]) {
 uint blocks=p[0]/16;if(i>=blocks*p[1])return;uint row=i/blocks,first=i%blocks*16;float4x4 v=weights16(w+ulong(row)*p[2],first,p[0]);device float4*dest=reinterpret_cast<device float4*>(out+ulong(row)*p[0]+first);for(uint j=0;j<4;++j)dest[j]=v[j];
}
inline float4 q8_four(device const uchar*p){return float4(char4(as_type<char>(p[0]),as_type<char>(p[1]),as_type<char>(p[2]),as_type<char>(p[3])));}
template<uint NR>
inline void q8_dense(device const uchar*w,device const float*x,device float*y,constant uint*p,uint2 group,uint lane,uint sg) {
 uint row0=group.x*(4*NR)+sg*NR;if(row0>=p[1])return;uint token=group.y;float4 sum=0;device const float*data=x+ulong(token)*p[0];
 for(uint at=16*lane;at<p[0];at+=512){device const float4*input=reinterpret_cast<device const float4*>(data+at);float4 a=input[0],b=input[1],c=input[2],d=input[3];
  #pragma unroll
  for(uint r=0;r<NR;++r){if(row0+r>=p[1])continue;device const uchar*block=w+(ulong(p[4])*p[1]+row0+r)*p[2]+(at/32)*34;device const uchar*q=block+2+at%32;
   sum[r]+=half_at(block)*(dot(a,q8_four(q))+dot(b,q8_four(q+4))+dot(c,q8_four(q+8))+dot(d,q8_four(q+12)));
  }
 }
 sum.x=simd_sum(sum.x);sum.y=simd_sum(sum.y);sum.z=simd_sum(sum.z);sum.w=simd_sum(sum.w);if(!lane)for(uint r=0;r<NR;++r)if(row0+r<p[1])y[ulong(token)*p[1]+row0+r]=sum[r];
}
kernel void q8_dense1(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {q8_dense<1>(w,x,y,p,group,lane,sg);}
kernel void q8_dense2(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {q8_dense<2>(w,x,y,p,group,lane,sg);}
kernel void q8_dense4(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {q8_dense<4>(w,x,y,p,group,lane,sg);}

template<uint NR>
inline void bf16_dense(device const uchar*w,device const float*x,device float*y,constant uint*p,uint2 group,uint lane,uint sg) {
 uint row0=group.x*(4*NR)+sg*NR;if(row0>=p[1])return;uint token=group.y;float4 sum=0;device const float*data=x+ulong(token)*p[0];
 for(uint at=4*lane;at<p[0];at+=128){float4 input=*reinterpret_cast<device const float4*>(data+at);
  #pragma unroll
  for(uint r=0;r<NR;++r){if(row0+r>=p[1])continue;device const ushort4*weight=reinterpret_cast<device const ushort4*>(w+(ulong(p[4])*p[1]+row0+r)*p[2]);
   sum[r]+=dot(input,as_type<float4>(uint4(weight[at/4])<<16));
  }
 }
 sum.x=simd_sum(sum.x);sum.y=simd_sum(sum.y);sum.z=simd_sum(sum.z);sum.w=simd_sum(sum.w);if(!lane)for(uint r=0;r<NR;++r)if(row0+r<p[1])y[ulong(token)*p[1]+row0+r]=sum[r];
}
kernel void bf16_dense1(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {bf16_dense<1>(w,x,y,p,group,lane,sg);}
kernel void bf16_dense2(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {bf16_dense<2>(w,x,y,p,group,lane,sg);}
kernel void bf16_dense4(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {bf16_dense<4>(w,x,y,p,group,lane,sg);}

// Each output reads the immutable input window; commit history in a later encoder.
kernel void convolution_parallel(device const float*x [[buffer(0)]],device const uchar*w [[buffer(1)]],device const float*history [[buffer(2)]],device float*y [[buffer(3)]],device float*snap [[buffer(4)]],constant uint*p [[buffer(5)]],uint i [[thread_position_in_grid]]) {
 if(i>=p[0]*p[5])return;uint t=i/p[0],c=i%p[0],taps=p[1],dil=p[2],hist=(taps-1)*dil;
 float input=x[i],v=input*weight_at(w+ulong(c)*p[4],p[3],taps-1);
 for(uint k=0;k<taps-1;++k){int at=int(t)-int(hist)+int(k*dil);float h=at>=0?x[ulong(at)*p[0]+c]:history[c*hist+uint(int(hist)+at)];v+=weight_at(w+ulong(c)*p[4],p[3],k)*h;}
 y[i]=v*sigmoid_f(v);
 if(t<p[6])for(uint j=0;j<hist;++j){int at=int(t)+1-int(hist)+int(j);snap[(ulong(t)*p[0]+c)*hist+j]=at>=0?x[ulong(at)*p[0]+c]:history[c*hist+uint(int(hist)+at)];}
}
kernel void convolution_history(device const float*x [[buffer(0)]],device float*history [[buffer(1)]],constant uint*p [[buffer(2)]],uint c [[thread_position_in_grid]]) {
 if(c>=p[0])return;uint hist=(p[1]-1)*p[2];
 for(uint j=0;j<hist;++j){int at=int(p[3])-int(hist)+int(j);history[c*hist+j]=at>=0?x[ulong(at)*p[0]+c]:history[c*hist+p[3]+j];}
}

// HC down and inject share one dispatch; preserve the selected BF16 reduction.
kernel void hc_down_inject(device const uchar*down [[buffer(0)]],device const uchar*inject [[buffer(1)]],device const float*x [[buffer(2)]],device float*lo [[buffer(3)]],device float*injection [[buffer(4)]],constant uint*p [[buffer(5)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint down_groups=p[1];
 bool is_inject=group.x>=down_groups;uint token=group.y;
 device const float*input=x+ulong(token)*p[0];
 // Count changes only the token grid, never the reduction topology.
  uint row=is_inject?group.x-down_groups:group.x;
  device const ushort4*w=reinterpret_cast<device const ushort4*>((is_inject?inject:down)+ulong(row)*(is_inject?p[3]:p[2]));
  float value=0;
  for(uint j=4*lane+128*sg;j<p[0];j+=512){float4 v=as_type<float4>(uint4(w[j/4])<<16);value+=v.x*input[j];value+=v.y*input[j+1];value+=v.z*input[j+2];value+=v.w*input[j+3];}
  value=simd_sum(value);threadgroup float partial[4];if(!lane)partial[sg]=value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(!lane&&!sg){float v=((partial[0]+partial[1])+partial[2])+partial[3];if(is_inject)injection[ulong(token)*p[4]+row]=v;else{v*=as_type<float>(p[6]);lo[ulong(token)*p[1]+row]=v*sigmoid_f(v);}}
}
kernel void hc_collapse_sigmoid(device const float*x [[buffer(0)]],device const float*g [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint i [[thread_position_in_grid]]) {
 if(i>=p[0]*p[2])return;uint t=i/p[0],j=i%p[0];float v=0;
 for(uint h=0;h<p[1];++h){uint k=(t*p[1]+h)*p[0]+j;float gate=sigmoid_f(g[k]);v+=x[k]*gate;}y[i]=v/p[1];
}

// One SIMD group computes the four HC gates for one output channel.
kernel void hc_up_collapse(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device const float*norm [[buffer(2)]],device float*y [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint j=group.x*4+sg,token=group.y;if(j>=p[0])return;float4 sums=0;
 device const float*input=x+ulong(token)*p[2];
 // Identical float4-dot accumulation for each actual token in count1..4.
  for(uint at=4*lane;at<p[2];at+=128){float4 v=*reinterpret_cast<device const float4*>(input+at);
   #pragma unroll
   for(uint h=0;h<4;++h){device const ushort4*row=reinterpret_cast<device const ushort4*>(w+ulong(h*p[0]+j)*p[3]);sums[h]+=dot(v,as_type<float4>(uint4(row[at/4])<<16));}
  }
 sums.x=simd_sum(sums.x);sums.y=simd_sum(sums.y);sums.z=simd_sum(sums.z);sums.w=simd_sum(sums.w);
 if(!lane){float value=0;for(uint h=0;h<4;++h)value+=norm[(ulong(token)*4+h)*p[0]+j]*sigmoid_f(sums[h]);y[ulong(token)*p[0]+j]=value/4.0f;}
}
