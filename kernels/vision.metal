// Vision encoder (Qwen3-VL style ViT) and image positions for the text model.
// LayerNorm with bias over rows of p[0]; one SIMD group per row.
kernel void vit_norm(device const float*x [[buffer(0)]],device const float*w [[buffer(1)]],device const float*b [[buffer(2)]],device float*y [[buffer(3)]],constant uint*p [[buffer(4)]],uint row [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
 uint n=p[0];device const float*r=x+ulong(row)*n;float s=0;for(uint j=lane;j<n;j+=32)s+=r[j];float mean=simd_sum(s)/float(n);float v=0;for(uint j=lane;j<n;j+=32){float z=r[j]-mean;v+=z*z;}float inv=rsqrt(simd_sum(v)/float(n)+as_type<float>(p[1]));for(uint j=lane;j<n;j+=32)y[ulong(row)*n+j]=(r[j]-mean)*inv*w[j]+b[j];
}
// y = gelu_tanh(x + b) with b of period p[1].
kernel void vit_bias_gelu(device const float*x [[buffer(0)]],device const float*b [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint i [[thread_position_in_grid]]){if(i>=p[0])return;float v=x[i]+b[i%p[1]];y[i]=0.5f*v*(1.0f+precise::tanh(0.7978845608028654f*v*(1.0f+0.044715f*v*v)));}
// y = r + x + b with b of period p[1].
kernel void vit_bias_add(device const float*r [[buffer(0)]],device const float*x [[buffer(1)]],device const float*b [[buffer(2)]],device float*y [[buffer(3)]],constant uint*p [[buffer(4)]],uint i [[thread_position_in_grid]]){if(i<p[0])y[i]=r[i]+x[i]+b[i%p[1]];}
// Split biased QKV rows into Q, K, V of p[1] tokens padded to p[2] rows; 2D RoPE on
// Q and K: the first half of each head rotates by the patch row, the second by its
// column, pairs (j, j + head/2), frequencies base^(-2f/(head/2)).
kernel void vit_qkv(device const float*x [[buffer(0)]],device const float*b [[buffer(1)]],device const uint*pos [[buffer(2)]],device float*q [[buffer(3)]],device float*k [[buffer(4)]],device float*v [[buffer(5)]],constant uint*p [[buffer(6)]],uint i [[thread_position_in_grid]]){
 uint width=p[0],d=p[3],t=i/width,c=i%width;if(t>=p[2])return;if(t>=p[1]){q[i]=0;k[i]=0;v[i]=0;return;}
 device const float*row=x+ulong(t)*3*width;uint e=c%d,base=c-e,half_d=d/2,j=e%half_d,quarter=half_d/2;
 float theta=float(pos[t*2+(j<quarter?0:1)])*pow(as_type<float>(p[4]),-2.0f*float(j%quarter)/float(half_d));float cs=cos(theta),sn=sin(theta);
 v[i]=row[2*width+c]+b[2*width+c];
 for(uint m=0;m<2;++m){device const float*r=row+m*width;device const float*bb=b+m*width;float lo=r[base+j]+bb[base+j],hi=r[base+j+half_d]+bb[base+j+half_d];float out=e<half_d?lo*cs-hi*sn:lo*sn+hi*cs;if(m==0)q[i]=out;else k[i]=out;}
}
// Non-causal attention over p[0] tokens (rows padded to a multiple of 8), p[1] heads
// of 72 dims, Q/K/V/out rows of p[2] floats. One SIMD group handles eight queries of
// one head with 8x8 matrix tiles and an online softmax.
kernel void vit_attention(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device float*out [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
 const uint D=72,T=9;uint n=p[0],stride=p[2],h=g.y,q0=g.x*8;float scale=as_type<float>(p[3]);
 threadgroup float s[64],diag[64],o[8*72],m[8],l[8];
 simdgroup_float8x8 Q[T],O[T];
 for(uint i=0;i<T;++i){simdgroup_load(Q[i],q+ulong(q0)*stride+h*D+i*8,stride);O[i]=make_filled_simdgroup_matrix<float,8,8>(0.0f);}
 if(lane<8){m[lane]=-INFINITY;l[lane]=0;}
 for(uint j=lane;j<64;j+=32)diag[j]=0;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 uint tiles=(n+7)/8;
 for(uint kt=0;kt<tiles;++kt){
  simdgroup_float8x8 S=make_filled_simdgroup_matrix<float,8,8>(0.0f),K;
  for(uint i=0;i<T;++i){simdgroup_load(K,k+ulong(kt*8)*stride+h*D+i*8,stride,ulong2(0,0),true);simdgroup_multiply_accumulate(S,Q[i],K,S);}
  simdgroup_store(S,s,8);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(lane<8){float mx=m[lane];for(uint c=0;c<8;++c){uint key=kt*8+c;float z=key<n?s[lane*8+c]*scale:-INFINITY;s[lane*8+c]=z;mx=max(mx,z);}
   float sum=0;for(uint c=0;c<8;++c){float e=exp(s[lane*8+c]-mx);s[lane*8+c]=e;sum+=e;}
   float alpha=exp(m[lane]-mx);l[lane]=l[lane]*alpha+sum;m[lane]=mx;diag[lane*9]=alpha;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_float8x8 P,A,V,R;simdgroup_load(P,s,8);simdgroup_load(A,diag,8);
  for(uint i=0;i<T;++i){simdgroup_load(V,v+ulong(kt*8)*stride+h*D+i*8,stride);simdgroup_multiply(R,A,O[i]);simdgroup_multiply_accumulate(O[i],P,V,R);}
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 for(uint i=0;i<T;++i)simdgroup_store(O[i],o+i*8,D);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint j=lane;j<8*D;j+=32){uint r=j/D,c=j%D;if(q0+r<n)out[ulong(q0+r)*stride+h*D+c]=o[j]/l[r];}
}
// Text RoPE with explicit per-row positions (t, h, w) and interleaved M-RoPE sections
// p[5..8]; x is p[4] rows of p[1] heads of p[0] dims, rotating the first p[2] dims.
kernel void rope_rows(device const float*x [[buffer(0)]],device const uint*pos [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint i [[thread_position_in_grid]]){
 uint width=p[0]*p[1];if(i>=width*p[4])return;uint j=i%p[0],r=p[2],base=i-j,row=i/width;if(j>=r){y[i]=x[i];return;}
 uint f=j%(r/2),axis=f%3==1&&f<3*p[6]?1:f%3==2&&f<3*p[7]?2:f%3==0&&f<3*p[5]?0:3;float position=axis<3?float(pos[row*3+axis]):0.0f;
 float theta=position*pow(as_type<float>(p[3]),-2.0f*float(f)/float(r));float c=cos(theta),s=sin(theta);y[i]=j<r/2?x[base+f]*c-x[base+f+r/2]*s:x[base+f]*s+x[base+f+r/2]*c;
}
