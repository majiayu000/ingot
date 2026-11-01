// Formats introduced by the Qwen3.8 IQ3_XXS mix: Q4_K (12), Q5_K (13),
// IQ2_XS (17) and IQ3_XXS (18). Block layouts and dequantization formulas follow
// ggml; see kernels/NOTICE. mix_weight_at, mix_weights16 and the vector/routed
// kernels evaluate each weight with the same F32 expression.

inline float mix_half(device const uchar*p){return float(*reinterpret_cast<device const half*>(p));}
// IQ2_XS (74 bytes) and IQ3_XXS (98 bytes) blocks are only 2-byte aligned.
inline uint mix_u16(device const uchar*p){return *reinterpret_cast<device const ushort*>(p);}
inline uint mix_u32(device const uchar*p){device const ushort*h=reinterpret_cast<device const ushort*>(p);return uint(h[0])|(uint(h[1])<<16);}
inline bool mix_type(uint t){return t==12||t==13||t==17||t==18;}
inline uint mix_block(uint t){return t==12?144:t==13?176:t==17?74:98;}
// ggml get_scale_min_k4: six-bit scale (x) and min (y) of 32-value sub-block j.
inline uint2 kquant_scale_min(device const uchar*s,uint j){
 return j<4?uint2(s[j]&63,s[j+4]&63):uint2((s[j+4]&15)|((s[j-4]>>6)<<4),(s[j+4]>>4)|((s[j]>>6)<<4));
}

inline float mix_weight_at(device const uchar*row,uint type,uint j){
 device const uchar*b=row+(j/256)*mix_block(type);uint k=j%256,g=k/32,l=(k%32)/8,v=k%8;
 if(type==12||type==13){ // d*scale*q - dmin*min; Q5_K adds bit g of qh as 16.
  uint2 sm=kquant_scale_min(b+4,g);
  uint q=(b[(type==12?16:48)+32*(k/64)+k%32]>>(4*(g%2)))&15;
  if(type==13)q|=((b[16+k%32]>>g)&1)<<4;
  return mix_half(b)*float(sm.x)*float(q)-mix_half(b+2)*float(sm.y);
 }
 if(type==17){ // 9-bit grid index and 7-bit sign index per 8 values, 4-bit scale per 16.
  uint q=mix_u16(b+2+2*(4*g+l));
  float grid=float((iq2xs_grid[q&511]>>(8*v))&255);
  float sign=(ksigns_iq2xs[q>>9]&(1<<v))?-1.0f:1.0f;
  return mix_half(b)*(0.5f+((b[66+g]>>(4*(l/2)))&15))*0.25f*grid*sign;
 }
 // IQ3_XXS: one 8-bit grid index per 4 values; scale and four sign indices per 32.
 uint aux=mix_u32(b+66+4*g);
 float grid=float((iq3xxs_grid[b[2+8*g+2*l+v/4]]>>(8*(v%4)))&255);
 float sign=(ksigns_iq2xs[(aux>>(7*l))&127]&(1<<v))?-1.0f:1.0f;
 return mix_half(b)*(0.5f+(aux>>28))*0.5f*grid*sign;
}

// Sixteen consecutive values from a 16-aligned column.
inline float4x4 mix_weights16(device const uchar*row,uint first,uint type){
 float4x4 r;device const uchar*b=row+(first/256)*mix_block(type);uint k=first%256,g=k/32,h=(k%32)/16;
 if(type==12||type==13){
  uint2 sm=kquant_scale_min(b+4,g);float d=mix_half(b)*float(sm.x),m=mix_half(b+2)*float(sm.y);
  uint l0=k%32;device const uchar*q=b+(type==12?16:48)+32*(k/64)+l0;
  for(uint i=0;i<4;++i){
   uint bits=(mix_u32(q+4*i)>>(4*(g%2)))&0x0f0f0f0fu;
   if(type==13)bits|=((mix_u32(b+16+l0+4*i)>>g)&0x01010101u)<<4;
   r[i]=d*float4(as_type<uchar4>(bits))-m;
  }
 } else if(type==17){
  float d=mix_half(b)*(0.5f+((b[66+g]>>(4*h))&15))*0.25f;
  for(uint t=0;t<2;++t){
   uint q=mix_u16(b+2+2*(4*g+2*h+t)),s=ksigns_iq2xs[q>>9];
   constant uchar4*grid=reinterpret_cast<constant uchar4*>(iq2xs_grid+(q&511));
   float4 lo=d*float4(grid[0]),hi=d*float4(grid[1]);
   r[2*t]=select(lo,-lo,(uint4(s)&uint4(1,2,4,8))!=uint4(0));
   r[2*t+1]=select(hi,-hi,(uint4(s)&uint4(16,32,64,128))!=uint4(0));
  }
 } else {
  uint aux=mix_u32(b+66+4*g);float d=mix_half(b)*(0.5f+(aux>>28))*0.5f;
  for(uint t=0;t<2;++t){
   uint l=2*h+t,s=ksigns_iq2xs[(aux>>(7*l))&127];
   float4 lo=d*float4(as_type<uchar4>(iq3xxs_grid[b[2+8*g+2*l]])),hi=d*float4(as_type<uchar4>(iq3xxs_grid[b[3+8*g+2*l]]));
   r[2*t]=select(lo,-lo,(uint4(s)&uint4(1,2,4,8))!=uint4(0));
   r[2*t+1]=select(hi,-hi,(uint4(s)&uint4(16,32,64,128))!=uint4(0));
  }
 }
 return r;
}

// Lane order lane, lane+32, ... as in dot_quant_rows; up to four token columns.
inline float4 mix_dot_rows(device const uchar*row,device const float*x,uint cols,uint lane,uint stride,uint count,uint type){
 float4 sum=0.0f;
 for(uint base=0;base<cols;base+=256)
  for(uint g=0;g<8;++g){
   uint j=base+lane+32*g;
   sum+=mix_weight_at(row,type,j)*float4(x[j],count>1?x[j+stride]:0.0f,count>2?x[j+2*stride]:0.0f,count>3?x[j+3*stride]:0.0f);
  }
 return sum;
}

// Q4_K/Q5_K projection for NT (one to four) tokens: two rows per SIMD group, eight
// per threadgroup. Sixteen lanes share a block; lane t owns columns 8*(t%4)..+7 of
// both sub-blocks in 64-column chunk t/4. Weights are decoded once for all tokens
// and every token keeps the NT=1 arithmetic, so counts one to four agree bitwise.
template<bool Five,uint NT>
inline void kquant_vector(device const uchar*w,device const float*x,device float*y,constant uint*p,uint2 group,uint lane,uint sg){
 uint row0=group.x*8+sg*2;if(row0>=p[1])return;
 uint ix=lane/16,c=(lane%16)/4,l0=(lane%4)*8;
 float2 sums[NT];
 for(uint t=0;t<NT;++t)sums[t]=0;
 for(uint block=ix;block<p[0]/256;block+=2){
  float4 a0[NT],a1[NT],b0[NT],b1[NT];float sa[NT],sb[NT];
  #pragma unroll
  for(uint t=0;t<NT;++t){
   device const float4*xb=reinterpret_cast<device const float4*>(x+ulong(t)*p[0]+block*256+64*c+l0);
   a0[t]=xb[0];a1[t]=xb[1];b0[t]=xb[8];b1[t]=xb[9];
   sa[t]=(a0[t].x+a0[t].y+a0[t].z+a0[t].w)+(a1[t].x+a1[t].y+a1[t].z+a1[t].w);
   sb[t]=(b0[t].x+b0[t].y+b0[t].z+b0[t].w)+(b1[t].x+b1[t].y+b1[t].z+b1[t].w);
  }
  #pragma unroll
  for(uint row=0;row<2;++row){
   if(row0+row>=p[1])continue;
   device const uchar*b=w+(ulong(p[4])*p[1]+row0+row)*p[2]+block*(Five?176:144);
   uint2 lo=kquant_scale_min(b+4,2*c),hi=kquant_scale_min(b+4,2*c+1);
   device const uchar*q=b+(Five?48:16)+32*c+l0;
   uint q0=mix_u32(q),q1=mix_u32(q+4);
   uint4 bits=uint4(q0&0x0f0f0f0fu,q1&0x0f0f0f0fu,(q0>>4)&0x0f0f0f0fu,(q1>>4)&0x0f0f0f0fu);
   if(Five){
    uint h0=mix_u32(b+16+l0),h1=mix_u32(b+16+l0+4);
    bits|=uint4((h0>>(2*c))&0x01010101u,(h1>>(2*c))&0x01010101u,(h0>>(2*c+1))&0x01010101u,(h1>>(2*c+1))&0x01010101u)<<4;
   }
   float4 wa0=float4(as_type<uchar4>(bits.x)),wa1=float4(as_type<uchar4>(bits.y)),wb0=float4(as_type<uchar4>(bits.z)),wb1=float4(as_type<uchar4>(bits.w));
   float d=mix_half(b),m=mix_half(b+2);
   #pragma unroll
   for(uint t=0;t<NT;++t){
    float da=dot(a0[t],wa0)+dot(a1[t],wa1),db=dot(b0[t],wb0)+dot(b1[t],wb1);
    sums[t][row]+=d*(float(lo.x)*da+float(hi.x)*db)-m*(float(lo.y)*sa[t]+float(hi.y)*sb[t]);
   }
  }
 }
 #pragma unroll
 for(uint t=0;t<NT;++t){
  float2 v=float2(simd_sum(sums[t].x),simd_sum(sums[t].y));
  if(!lane)for(uint row=0;row<min(2u,p[1]-row0);++row)y[ulong(t)*p[1]+row0+row]=v[row];
 }
}
#define KQUANT_VECTOR(name,five,nt) kernel void name(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {kquant_vector<five,nt>(w,x,y,p,group,lane,sg);}
KQUANT_VECTOR(q4k_vector,false,1) KQUANT_VECTOR(q4k_vector2,false,2) KQUANT_VECTOR(q4k_vector3,false,3) KQUANT_VECTOR(q4k_vector4,false,4)
KQUANT_VECTOR(q5k_vector,true,1) KQUANT_VECTOR(q5k_vector2,true,2) KQUANT_VECTOR(q5k_vector3,true,3) KQUANT_VECTOR(q5k_vector4,true,4)

// IQ2_XS/IQ3_XXS/IQ3_S routed rows in project_routed_batch geometry: four rows per SIMD
// group, sixteen per threadgroup, and each lane owns 32-value quant groups at a
// 1024-column stride, sharing the group scale and sign word across its rows.
template<uint Type>
inline void routed_mix(device const uchar*w,device const float*x,device const uint*ids,device float*y,constant uint*p,uint2 group,uint lane,uint sg){
 uint row=group.x*16+sg*4;if(row>=p[1])return;
 uint pair=group.y,input=p[5]?pair:pair/p[4],valid=min(4u,p[1]-row);
 device const uchar*rows=w+(ulong(ids[pair])*p[1]+row)*p[2];
 device const float*data=x+ulong(input)*p[0];
 float4 sum=0.0f;
 for(uint at=lane*32;at<p[0];at+=32*32){
  float4 values[8];
  #pragma unroll
  for(uint j=0;j<8;++j)values[j]=*reinterpret_cast<device const float4*>(data+at+4*j);
  uint g=(at%256)/32;
  #pragma unroll
  for(uint r=0;r<4;++r){
   if(r>=valid)continue;
   device const uchar*b=rows+ulong(r)*p[2]+(at/256)*(Type==17?74:Type==18?98:110);
   float sub=0.0f;
   if(Type==21){
    float d=mix_half(b)*(1+2*((b[106+g/2]>>(4*(g%2)))&15));
    #pragma unroll
    for(uint l=0;l<4;++l){
     uint s=b[74+4*g+l],qh=b[66+g];
     float4 lo=float4(as_type<uchar4>(iq3s_grid[b[2+8*g+2*l]|((qh<<(8-2*l))&256)]));
     float4 hi=float4(as_type<uchar4>(iq3s_grid[b[3+8*g+2*l]|((qh<<(7-2*l))&256)]));
     lo=select(lo,-lo,(uint4(s)&uint4(1,2,4,8))!=uint4(0));
     hi=select(hi,-hi,(uint4(s)&uint4(16,32,64,128))!=uint4(0));
     sub+=dot(values[2*l],lo)+dot(values[2*l+1],hi);
    }
    sum[r]+=d*sub;
   } else if(Type==17){
    #pragma unroll
    for(uint l=0;l<4;++l){
     uint q=mix_u16(b+2+2*(4*g+l)),s=ksigns_iq2xs[q>>9];
     constant uchar4*grid=reinterpret_cast<constant uchar4*>(iq2xs_grid+(q&511));
     float4 lo=float4(grid[0]),hi=float4(grid[1]);
     lo=select(lo,-lo,(uint4(s)&uint4(1,2,4,8))!=uint4(0));
     hi=select(hi,-hi,(uint4(s)&uint4(16,32,64,128))!=uint4(0));
     sub+=(0.5f+((b[66+g]>>(4*(l/2)))&15))*(dot(values[2*l],lo)+dot(values[2*l+1],hi));
    }
    sum[r]+=mix_half(b)*sub;
   } else {
    uint aux=mix_u32(b+66+4*g);
    #pragma unroll
    for(uint l=0;l<4;++l){
     uint s=ksigns_iq2xs[(aux>>(7*l))&127];
     float4 lo=float4(as_type<uchar4>(iq3xxs_grid[b[2+8*g+2*l]])),hi=float4(as_type<uchar4>(iq3xxs_grid[b[3+8*g+2*l]]));
     lo=select(lo,-lo,(uint4(s)&uint4(1,2,4,8))!=uint4(0));
     hi=select(hi,-hi,(uint4(s)&uint4(16,32,64,128))!=uint4(0));
     sub+=dot(values[2*l],lo)+dot(values[2*l+1],hi);
    }
    sum[r]+=mix_half(b)*(0.5f+(aux>>28))*sub;
   }
  }
 }
 sum*=Type==17?0.25f:Type==18?0.5f:1.0f;
 sum.x=simd_sum(sum.x);sum.y=simd_sum(sum.y);sum.z=simd_sum(sum.z);sum.w=simd_sum(sum.w);
 if(!lane)for(uint r=0;r<valid;++r)y[ulong(pair)*p[1]+row+r]=sum[r];
}
kernel void routed_iq2xs(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device const uint*ids [[buffer(2)]],device float*y [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {routed_mix<17>(w,x,ids,y,p,group,lane,sg);}
kernel void routed_iq3xxs(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device const uint*ids [[buffer(2)]],device float*y [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {routed_mix<18>(w,x,ids,y,p,group,lane,sg);}
kernel void routed_iq3s(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device const uint*ids [[buffer(2)]],device float*y [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {routed_mix<21>(w,x,ids,y,p,group,lane,sg);}

// IQ4_NL routed rows (expert down projections): four rows per SIMD group; a lane
// owns eight values of a 32-value block, four low and four high nibbles.
kernel void routed_iq4nl(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device const uint*ids [[buffer(2)]],device float*y [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {
 uint row=group.x*16+sg*4;if(row>=p[1])return;
 uint pair=group.y,input=p[5]?pair:pair/p[4],valid=min(4u,p[1]-row);
 device const uchar*rows=w+(ulong(ids[pair])*p[1]+row)*p[2];
 device const float*data=x+ulong(input)*p[0];
 float4 sum=0.0f;
 for(uint unit=lane;unit<p[0]/8;unit+=32){
  uint block=unit/4,q=unit%4;
  float4 xl=*reinterpret_cast<device const float4*>(data+block*32+4*q),xh=*reinterpret_cast<device const float4*>(data+block*32+16+4*q);
  #pragma unroll
  for(uint r=0;r<4;++r){
   if(r>=valid)continue;
   device const uchar*b=rows+ulong(r)*p[2]+block*18;
   uint bytes=mix_u32(b+2+4*q);
   float4 lo,hi;
   #pragma unroll
   for(uint j=0;j<4;++j){lo[j]=kvalues_iq4nl[(bytes>>(8*j))&15];hi[j]=kvalues_iq4nl[(bytes>>(8*j+4))&15];}
   sum[r]+=mix_half(b)*(dot(xl,lo)+dot(xh,hi));
  }
 }
 sum.x=simd_sum(sum.x);sum.y=simd_sum(sum.y);sum.z=simd_sum(sum.z);sum.w=simd_sum(sum.w);
 if(!lane)for(uint r=0;r<valid;++r)y[ulong(pair)*p[1]+row+r]=sum[r];
}
