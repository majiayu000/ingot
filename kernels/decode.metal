// hc_add followed by the next HC block's RMS norm. Each row keeps rms_wide's
// 128-thread strided sums and two-level SIMD reductions, so y and yn match
// hc_add then rms_wide bit for bit. Two rows per 256-thread group.
// p: [width, hc rows, norm weight length, eps bits].
kernel void hc_add_rms(device const float*x [[buffer(0)]],device const float*b [[buffer(1)]],device const float*g [[buffer(2)]],device const float*w [[buffer(3)]],device float*y [[buffer(4)]],device float*yn [[buffer(5)]],constant uint*p [[buffer(6)]],uint group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint n=p[0],row=group*2+tid/128,t=tid%128;
 threadgroup float partial[8];threadgroup float inv[2];
 float sum=0;
 if(row<p[1])for(uint j=t;j<n;j+=128){uint i=row*n+j;float v=x[i]+b[j]*2.0f*sigmoid_f(g[row]/p[1]);y[i]=v;sum+=v*v;}
 sum=simd_sum(sum);if(!lane)partial[sg]=sum;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(sg%4==0){float total=simd_sum(lane<4?partial[sg+lane]:0.0f);if(!lane)inv[sg/4]=rsqrt(total/float(n)+as_type<float>(p[3]));}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(row<p[1])for(uint j=t;j<n;j+=128){uint i=row*n+j;yn[i]=y[i]*inv[tid/128]*w[i%p[2]];}
}

// IQ3_S projection for one to four tokens, laid out like iq4_vector: two rows
// per SIMD group, eight per threadgroup, one 16-value half group per lane.
// 110-byte blocks are only 2-byte aligned, so packed fields use ushort loads.
// Every token count shares this per-token order, keeping AR and MTP verify equal.
kernel void iq3s_vector(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint row0=group.x*8+sg*2;if(row0>=p[1])return;
 uint token=group.y,ix=lane/16,g=(lane%16)/2,h=lane%2;
 float2 sums=0;
 for(uint block=ix;block<p[0]/256;block+=2) {
  device const float4*input=reinterpret_cast<device const float4*>(x+ulong(token)*p[0]+block*256+g*32+h*16);
  float4 xs[4]={input[0],input[1],input[2],input[3]};
  #pragma unroll
  for(uint row=0;row<2;++row) {
   if(row0+row>=p[1])continue;
   device const uchar*b=w+(ulong(p[4])*p[1]+row0+row)*p[2]+block*110;
   device const ushort*qs=reinterpret_cast<device const ushort*>(b+2+g*8+h*4);
   uint bytes=uint(qs[0])|(uint(qs[1])<<16),qh=b[66+g];
   uint signs=*reinterpret_cast<device const ushort*>(b+74+g*4+h*2);
   float4 acc=0;
   #pragma unroll
   for(uint i=0;i<4;++i) {
    uint m=h*4+i,index=((bytes>>(8*i))&255)|(((qh>>m)&1)<<8);
    float4 grid=float4(as_type<uchar4>(iq3s_grid[index]));
    uint4 bits=(uint4(signs>>(4*i))&uint4(1,2,4,8));
    acc+=xs[i]*select(grid,-grid,bits!=uint4(0));
   }
   float d=half_at(b)*(1+2*((b[106+g/2]>>(4*(g%2)))&15));
   sums[row]+=d*(acc.x+acc.y+acc.z+acc.w);
  }
 }
 sums.x=simd_sum(sums.x);sums.y=simd_sum(sums.y);
 if(!lane)for(uint row=0;row<min(2u,p[1]-row0);++row)y[ulong(token)*p[1]+row0+row]=sums[row];
}

inline uint packed4(device const uchar*p) {
 device const ushort*h=reinterpret_cast<device const ushort*>(p);return uint(h[0])|(uint(h[1])<<16);
}

// Q6_K projection for one to four tokens: per row, sixteen lanes own a block
// half each (four consecutive l by four quarters t); blocks are 2-byte aligned.
inline void q6k_vector_impl(device const uchar*w,device const float*x,device float*y,constant uint*p,uint2 group,uint lane,uint sg) {
 uint row0=group.x*8+sg*2;if(row0>=p[1])return;
 uint token=group.y,ix=lane/16,n=(lane%16)/8,l0=(lane%8)*4;
 float2 sums=0;
 for(uint block=ix;block<p[0]/256;block+=2) {
  device const float*xb=x+ulong(token)*p[0]+block*256+n*128+l0;
  float4 xs[4];
  #pragma unroll
  for(uint t=0;t<4;++t)xs[t]=*reinterpret_cast<device const float4*>(xb+32*t);
  #pragma unroll
  for(uint row=0;row<2;++row) {
   if(row0+row>=p[1])continue;
   device const uchar*b=w+(ulong(p[4])*p[1]+row0+row)*p[2]+block*210;
   uint lo0=packed4(b+n*64+l0),lo1=packed4(b+n*64+32+l0),hi=packed4(b+128+n*32+l0);
   device const uchar*sc=b+192+n*8+l0/16;
   float sum=0;
   #pragma unroll
   for(uint t=0;t<4;++t) {
    uint lo=((t%2)?lo1:lo0)>>(t>=2?4:0);
    int4 q=int4(as_type<uchar4>((lo&0x0f0f0f0fu)|(((hi>>(2*t))&0x03030303u)<<4)))-32;
    float4 v=xs[t]*float4(q);
    sum+=float(as_type<char>(sc[2*t]))*(v.x+v.y+v.z+v.w);
   }
   sums[row]+=half_at(b+208)*sum;
  }
 }
 sums.x=simd_sum(sums.x);sums.y=simd_sum(sums.y);
 if(!lane)for(uint row=0;row<min(2u,p[1]-row0);++row)y[ulong(token)*p[1]+row0+row]=sums[row];
}
kernel void q6k_vector(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {q6k_vector_impl(w,x,y,p,group,lane,sg);}

// IQ4_NL projection for one to four tokens: two lanes per 32-value block, each
// owning eight packed bytes (sixteen values); blocks are 2-byte aligned.
kernel void iq4nl_vector(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint row0=group.x*8+sg*2;if(row0>=p[1])return;
 uint token=group.y,ix=lane/2,h=lane%2;
 float2 sums=0;
 for(uint block=ix;block<p[0]/32;block+=16) {
  device const float*xb=x+ulong(token)*p[0]+block*32+h*8;
  float4 x0=*reinterpret_cast<device const float4*>(xb),x1=*reinterpret_cast<device const float4*>(xb+4);
  float4 x2=*reinterpret_cast<device const float4*>(xb+16),x3=*reinterpret_cast<device const float4*>(xb+20);
  #pragma unroll
  for(uint row=0;row<2;++row) {
   if(row0+row>=p[1])continue;
   device const uchar*b=w+(ulong(p[4])*p[1]+row0+row)*p[2]+block*18;
   uint q0=packed4(b+2+h*8),q1=packed4(b+6+h*8);
   float4 l0,l1,h0,h1;
   #pragma unroll
   for(uint j=0;j<4;++j) {
    l0[j]=kvalues_iq4nl[(q0>>(8*j))&15];h0[j]=kvalues_iq4nl[(q0>>(8*j+4))&15];
    l1[j]=kvalues_iq4nl[(q1>>(8*j))&15];h1[j]=kvalues_iq4nl[(q1>>(8*j+4))&15];
   }
   float4 acc=(x0*l0+x1*l1)+(x2*h0+x3*h1);
   sums[row]+=half_at(b)*(acc.x+acc.y+acc.z+acc.w);
  }
 }
 sums.x=simd_sum(sums.x);sums.y=simd_sum(sums.y);
 if(!lane)for(uint row=0;row<min(2u,p[1]-row0);++row)y[ulong(token)*p[1]+row0+row]=sums[row];
}
