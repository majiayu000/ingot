// Generic transformer helpers (Llama / Qwen2 / Qwen3 families).
kernel void add_rows(device const float*a [[buffer(0)]],device const float*b [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint i [[thread_position_in_grid]]){if(i<p[0])y[i]=a[i]+b[i];}
// y = x + bias, with the bias broadcast over rows of width p[1].
kernel void bias_rows(device const float*x [[buffer(0)]],device const float*b [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint i [[thread_position_in_grid]]){if(i<p[0])y[i]=x[i]+b[i%p[1]];}
// RoPE for p[5] tokens at positions p[3]+t. p: head dim, heads, rotated dims,
// position, base bits, tokens, NeoX (split-half) or adjacent pairs, use factors.
// Frequency factors (Llama 3.1 rope_freqs) divide each pair's angle.
kernel void rope_generic(device const float*x [[buffer(0)]],device const float*factors [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint i [[thread_position_in_grid]]){
 uint width=p[0]*p[1];if(i>=width*p[5])return;uint j=i%p[0],r=p[2],base=i-j;
 if(j>=r){y[i]=x[i];return;}
 bool neox=p[6]!=0;uint f=neox?j%(r/2):j/2;uint a=neox?base+f:base+(j&~1u);uint b=neox?a+r/2:a+1;
 float theta=float(p[3]+i/width)*pow(as_type<float>(p[4]),-2.0f*float(f)/float(r));if(p[7])theta/=factors[f];
 float c=cos(theta),s=sin(theta);bool first=neox?j<r/2:(j&1u)==0;
 y[i]=first?x[a]*c-x[b]*s:x[a]*s+x[b]*c;
}
// Weighted sum of each token's top-k expert outputs: p = width, k, tokens.
kernel void moe_sum(device const float*routed [[buffer(0)]],device const float*w [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint i [[thread_position_in_grid]]){if(i>=p[0]*p[2])return;uint t=i/p[0],j=i%p[0];float s=0;for(uint e=0;e<p[1];++e)s+=routed[(ulong(t)*p[1]+e)*p[0]+j]*w[t*p[1]+e];y[i]=s;}

// Flash-style causal prefill attention (same algorithm as attention_prefill_tiles
// in qwen.metal) for head dimension D, scale 1/sqrt(D), no selection mask.
// p: head dim, heads, kv heads, length, position of the first row, rows,
// sliding window (0: none; key j is visible to query q when q - j < window).
template<bool Tail, uint D>
inline void flash_prefill_impl(device const float*q, device const float*k, device const float*v,
 device float*out, constant uint*p, uint2 group, uint tid, uint lane, uint sg,
 threadgroup float*sq, threadgroup float*tail, threadgroup float2*scores2, threadgroup float*rowf, threadgroup float*diag) {
    constexpr uint Q=8,R=1,J=D/32;const float scale=rsqrt(float(D));
    uint first=group.x*Q,h=group.y,kh=h/(p[1]/p[2]),stride=p[2]*D;
    threadgroup float*scores=reinterpret_cast<threadgroup float*>(scores2);
    for(uint i=tid;i<Q*D;i+=128) {
        uint t=first+i/D;
        sq[i]=t<p[5]?q[(ulong(t)*p[1]+h)*D+i%D]:0;
    }
    float maximum[Q/4],denominator[Q/4];
    for(uint j=0;j<Q/4;++j){maximum[j]=-INFINITY;denominator[j]=0;}
    simdgroup_float8x8 accum[R][J];
    #pragma unroll
    for(uint r=0;r<R;++r)for(uint j=0;j<J;++j)accum[r][j]=make_filled_simdgroup_matrix<float,8>(0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint end=min(p[3],p[4]+min(first+Q,p[5])),low=p[4]+first+1;
    for(uint base=p[6]&&low>p[6]?(low-p[6])/64*64:0;base<end;base+=64) {
        if(Tail && base+64>p[3]) {
            for(uint i=tid;i<8*D;i+=128) {
                uint pos=(p[3]/8)*8+i/D;
                tail[i]=pos<p[3]?k[ulong(pos)*stride+kh*D+i%D]:0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        #pragma unroll
        for(uint c=0;c<2;++c) {
            uint column=8*(sg+4*c);
            simdgroup_float8x8 total[R];
            for(uint r=0;r<R;++r)total[r]=make_filled_simdgroup_matrix<float,8>(0.0f);
            for(uint j=0;j<D;j+=8) {
                simdgroup_float8x8 mq,mk=make_filled_simdgroup_matrix<float,8>(0.0f);
                if(!Tail || base+column+8<=p[3]) simdgroup_load(mk,k+ulong(base+column)*stride+kh*D+j,stride,0,true);
                else if(base+column<p[3]) simdgroup_load(mk,tail+j,D,0,true);
                #pragma unroll
                for(uint r=0;r<R;++r){simdgroup_load(mq,sq+r*8*D+j,D,0,false);simdgroup_multiply_accumulate(total[r],mq,mk,total[r]);}
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
                float2 score=scores2[qr*32+lane]*scale;
                uint pos=base+2*lane,qp=p[4]+t;
                if(pos>qp||(p[6]&&qp-pos>=p[6]))score.x=-INFINITY;
                if(pos+1>qp||(p[6]&&qp-pos-1>=p[6]))score.y=-INFINITY;
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
            for(uint i=tid;i<8*D;i+=128) {
                uint pos=(p[3]/8)*8+i/D;
                tail[i]=pos<p[3]?v[ulong(pos)*stride+kh*D+i%D]:0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for(uint r=0;r<R;++r) {
            simdgroup_float8x8 dg;simdgroup_load(dg,diag+64*r,8,0,false);
            #pragma unroll
            for(uint j=0;j<J;++j)simdgroup_multiply(accum[r][j],dg,accum[r][j]);
        }
        for(uint c=0;c<64;c+=8) {
            simdgroup_float8x8 prob[R];
            #pragma unroll
            for(uint r=0;r<R;++r)simdgroup_load(prob[r],scores+r*8*64+c,64,0,false);
            #pragma unroll
            for(uint j=0;j<J;++j) {
                simdgroup_float8x8 value=make_filled_simdgroup_matrix<float,8>(0.0f);
                if(!Tail || base+c+8<=p[3]) simdgroup_load(value,v+ulong(base+c)*stride+kh*D+8*(sg+4*j),stride,0,false);
                else if(base+c<p[3]) simdgroup_load(value,tail+8*(sg+4*j),D,0,false);
                #pragma unroll
                for(uint r=0;r<R;++r)simdgroup_multiply_accumulate(accum[r][j],prob[r],value,accum[r][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for(uint r=0;r<R;++r)for(uint j=0;j<J;++j)simdgroup_store(accum[r][j],sq+r*8*D+8*(sg+4*j),D,0,false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for(uint j=0;j<Q/4;++j) {
        uint qr=sg+4*j,t=first+qr;
        if(t<p[5])for(uint col=lane;col<D;col+=32)
            out[(ulong(t)*p[1]+h)*D+col]=denominator[j]>0?sq[qr*D+col]/denominator[j]:0;
    }
}

kernel void flash_prefill_64(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device float*out [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {threadgroup float sq[8*64],tail[1],rowf[8],diag[64];threadgroup float2 scores2[256];flash_prefill_impl<false,64>(q,k,v,out,p,group,tid,lane,sg,sq,tail,scores2,rowf,diag);}
kernel void flash_prefill_64_tail(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device float*out [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {threadgroup float sq[8*64],tail[512],rowf[8],diag[64];threadgroup float2 scores2[256];flash_prefill_impl<true,64>(q,k,v,out,p,group,tid,lane,sg,sq,tail,scores2,rowf,diag);}
kernel void flash_prefill_96(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device float*out [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {threadgroup float sq[8*96],tail[1],rowf[8],diag[64];threadgroup float2 scores2[256];flash_prefill_impl<false,96>(q,k,v,out,p,group,tid,lane,sg,sq,tail,scores2,rowf,diag);}
kernel void flash_prefill_96_tail(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device float*out [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {threadgroup float sq[8*96],tail[768],rowf[8],diag[64];threadgroup float2 scores2[256];flash_prefill_impl<true,96>(q,k,v,out,p,group,tid,lane,sg,sq,tail,scores2,rowf,diag);}
kernel void flash_prefill_128(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device float*out [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {threadgroup float sq[8*128],tail[1],rowf[8],diag[64];threadgroup float2 scores2[256];flash_prefill_impl<false,128>(q,k,v,out,p,group,tid,lane,sg,sq,tail,scores2,rowf,diag);}
kernel void flash_prefill_128_tail(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device float*out [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {threadgroup float sq[8*128],tail[1024],rowf[8],diag[64];threadgroup float2 scores2[256];flash_prefill_impl<true,128>(q,k,v,out,p,group,tid,lane,sg,sq,tail,scores2,rowf,diag);}
kernel void flash_prefill_256(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device float*out [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {threadgroup float sq[8*256],tail[1],rowf[8],diag[64];threadgroup float2 scores2[256];flash_prefill_impl<false,256>(q,k,v,out,p,group,tid,lane,sg,sq,tail,scores2,rowf,diag);}
kernel void flash_prefill_256_tail(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device float*out [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {threadgroup float sq[8*256],tail[2048],rowf[8],diag[64];threadgroup float2 scores2[256];flash_prefill_impl<true,256>(q,k,v,out,p,group,tid,lane,sg,sq,tail,scores2,rowf,diag);}
// No tail variant: its key tile would not fit in threadgroup memory beside the
// query tile, so rows past the last whole 64-key tile use dense attention.
kernel void flash_prefill_512(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device float*out [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]) {threadgroup float sq[8*512],tail[1],rowf[8],diag[64];threadgroup float2 scores2[256];flash_prefill_impl<false,512>(q,k,v,out,p,group,tid,lane,sg,sq,tail,scores2,rowf,diag);}
// Routed decode projections for resident K-quant experts: the fast dense
// vector kernels with weight, input and output offset by each pair's expert.
// p: cols, rows, stride, kind, 0, per-pair input, top-k. Grid: rows/8 × pairs.
kernel void routed_q4k(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device const uint*ids [[buffer(2)]],device float*y [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]){
 uint pair=group.y,input=p[5]?pair:pair/p[6];
 kquant_vector<false,1>(w+ulong(ids[pair])*p[1]*p[2],x+ulong(input)*p[0],y+ulong(pair)*p[1],p,uint2(group.x,0),lane,sg);
}
kernel void routed_q6k(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device const uint*ids [[buffer(2)]],device float*y [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]){
 uint pair=group.y,input=p[5]?pair:pair/p[6];
 q6k_vector_impl(w+ulong(ids[pair])*p[1]*p[2],x+ulong(input)*p[0],y+ulong(pair)*p[1],p,uint2(group.x,0),lane,sg);
}
// Q4_0 matrix-vector for 1-4 tokens: each SIMD group computes two rows, lanes
// stride over 32-element blocks. p: cols, rows, stride, kind, expert, tokens.
kernel void q4_0_vector(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]){
 uint row0=group.x*8+sg*2;if(row0>=p[1])return;
 uint nr=min(2u,p[1]-row0),count=p[5];
 float acc[4][2]={{0,0},{0,0},{0,0},{0,0}};
 for(uint b=lane;b<p[0]/32;b+=32){
  device const uchar*blk[2];float d[2];uchar4 q[2][4];
  for(uint r=0;r<nr;++r){
   blk[r]=w+(ulong(p[4])*p[1]+row0+r)*p[2]+b*18;d[r]=half_at(blk[r]);
   for(uint i=0;i<4;++i)q[r][i]=uchar4(blk[r][2+4*i],blk[r][3+4*i],blk[r][4+4*i],blk[r][5+4*i]);
  }
  for(uint t=0;t<count;++t){
   device const float4*xv=reinterpret_cast<device const float4*>(x+ulong(t)*p[0]+b*32);
   float4 xs[8];for(uint i=0;i<8;++i)xs[i]=xv[i];
   for(uint r=0;r<nr;++r){
    float s=0;
    for(uint i=0;i<4;++i){
     float4 lo=float4(q[r][i]&uchar4(15))-8.0f,hi=float4(q[r][i]>>uchar4(4))-8.0f;
     s+=dot(lo,xs[i])+dot(hi,xs[4+i]);
    }
    acc[t][r]+=d[r]*s;
   }
  }
 }
 for(uint t=0;t<count;++t)for(uint r=0;r<nr;++r){float v=simd_sum(acc[t][r]);if(!lane)y[ulong(t)*p[1]+row0+r]=v;}
}

// Matrix-vector kernels for the generic path, adapted from ggml-org/llama.cpp
// (MIT) ggml/src/ggml-metal/kernels/mul_mv.metal at 9b56ccf:
// kernel_mul_mv_q4_K_f32_impl, kernel_mul_mv_q6_K_f32_impl and
// mul_vec_q_n_f32_impl (Q4_0). Each SIMD group computes NR rows; w, x and y
// arrive offset to the expert, token and output row 0. p[0] cols, p[1] rows,
// p[2] row bytes.
template<uint NR>
inline void mv_q4k(device const uchar*w,device const float*x,constant uint*p,uint row0,ushort lane,thread float*out){
 const short ix=lane/8,it=lane%8,iq=it/4,ir=it%4;
 device const ushort*rows[NR];
 for(uint r=0;r<NR;++r)rows[r]=reinterpret_cast<device const ushort*>(w+ulong(min(row0+r,p[1]-1))*p[2]);
 float yl[16],yh[16],sumf[NR]={0};
 ushort sc16[4];thread const uchar*sc8=reinterpret_cast<thread const uchar*>(sc16);
 device const float*y4=x+ix*256+64*iq+8*ir;
 for(uint ib=ix;ib<p[0]/256;ib+=4){
  float4 sumy=0;
  for(short i=0;i<8;++i){
   yl[i]=y4[i];sumy[0]+=yl[i];yl[i+8]=y4[i+32];sumy[1]+=yl[i+8];
   yh[i]=y4[i+128];sumy[2]+=yh[i];yh[i+8]=y4[i+160];sumy[3]+=yh[i+8];
  }
  for(uint row=0;row<NR;++row){
   device const ushort*b=rows[row]+ib*72,*sc=b+2+iq,*q1=b+8+16*iq+4*ir,*q2=q1+32;
   sc16[0]=sc[0]&0x3f3f;sc16[1]=sc[2]&0x3f3f;
   sc16[2]=((sc[4]>>0)&0x0f0f)|((sc[0]&0xc0c0)>>2);sc16[3]=((sc[4]>>4)&0x0f0f)|((sc[2]&0xc0c0)>>2);
   float4 acc1=0,acc2=0;
   #pragma unroll
   for(short i=0;i<4;++i){
    acc1[0]+=yl[2*i+0]*(q1[i]&0x000F);acc1[1]+=yl[2*i+1]*(q1[i]&0x0F00);
    acc1[2]+=yl[2*i+8]*(q1[i]&0x00F0);acc1[3]+=yl[2*i+9]*(q1[i]&0xF000);
    acc2[0]+=yh[2*i+0]*(q2[i]&0x000F);acc2[1]+=yh[2*i+1]*(q2[i]&0x0F00);
    acc2[2]+=yh[2*i+8]*(q2[i]&0x00F0);acc2[3]+=yh[2*i+9]*(q2[i]&0xF000);
   }
   float d=as_type<half>(b[0]),m=as_type<half>(b[1]);
   sumf[row]+=d*((acc1[0]+1.f/256.f*acc1[1])*sc8[0]+(acc1[2]+1.f/256.f*acc1[3])*sc8[1]*1.f/16.f+
                 (acc2[0]+1.f/256.f*acc2[1])*sc8[4]+(acc2[2]+1.f/256.f*acc2[3])*sc8[5]*1.f/16.f)-
              m*(sumy[0]*sc8[2]+sumy[1]*sc8[3]+sumy[2]*sc8[6]+sumy[3]*sc8[7]);
  }
  y4+=4*256;
 }
 for(uint row=0;row<NR;++row)out[row]=simd_sum(sumf[row]);
}
template<uint NR>
inline void mv_q6k(device const uchar*w,device const float*x,constant uint*p,uint row0,ushort lane,thread float*out){
 const short tid=lane/2,ix=lane%2,ip=tid/8,il=tid%8,l0=4*il,is=8*ip+l0/16;
 device const uchar*rows[NR];
 for(uint r=0;r<NR;++r)rows[r]=w+ulong(min(row0+r,p[1]-1))*p[2];
 float yl[16],sumf[NR]={0};
 for(uint i=ix;i<p[0]/256;i+=2){
  device const float*yy=x+i*256+128*ip+l0;
  for(short l=0;l<4;++l){yl[4*l]=yy[l];yl[4*l+1]=yy[l+32];yl[4*l+2]=yy[l+64];yl[4*l+3]=yy[l+96];}
  for(uint row=0;row<NR;++row){
   device const uchar*b=rows[row]+i*210,*q1=b+64*ip+l0,*q2=q1+32,*qh=b+128+32*ip+l0;
   device const char*sc=reinterpret_cast<device const char*>(b+192+is);
   float4 sums=0;
   #pragma unroll
   for(short l=0;l<4;++l){
    sums[0]+=yl[4*l+0]*(float((q1[l]&0xF)|((qh[l]&0x03)<<4))-32);
    sums[1]+=yl[4*l+1]*(float((q2[l]&0xF)|((qh[l]&0x0C)<<2))-32);
    sums[2]+=yl[4*l+2]*(float((q1[l]>>4)|((qh[l]&0x30)))-32);
    sums[3]+=yl[4*l+3]*(float((q2[l]>>4)|((qh[l]&0xC0)>>2))-32);
   }
   sumf[row]+=float(as_type<half>(*reinterpret_cast<device const ushort*>(b+208)))*(sums[0]*sc[0]+sums[1]*sc[2]+sums[2]*sc[4]+sums[3]*sc[6]);
  }
 }
 for(uint row=0;row<NR;++row)out[row]=simd_sum(sumf[row]);
}
template<uint NR>
inline void mv_q4_0(device const uchar*w,device const float*x,constant uint*p,uint row0,ushort lane,thread float*out){
 const short ix=lane/2,il=(lane%2)*8;
 device const ushort*rows[NR];
 for(uint r=0;r<NR;++r)rows[r]=reinterpret_cast<device const ushort*>(w+ulong(min(row0+r,p[1]-1))*p[2]);
 float yl[16],sumf[NR]={0};
 device const float*yb=x+ix*32+il;
 for(uint ib=ix;ib<p[0]/32;ib+=16){
  float sumy=0;
  #pragma unroll
  for(short i=0;i<8;i+=2){
   sumy+=yb[i]+yb[i+1]+yb[i+16]+yb[i+17];
   yl[i]=yb[i];yl[i+1]=yb[i+1]/256.f;yl[i+8]=yb[i+16]/16.f;yl[i+9]=yb[i+17]/4096.f;
  }
  for(uint row=0;row<NR;++row){
   device const ushort*b=rows[row]+ib*9,*qs=b+1+il/2;
   float acc[4]={0,0,0,0};
   #pragma unroll
   for(short i=0;i<8;i+=2){
    acc[0]+=yl[i]*(qs[i/2]&0x000F);acc[1]+=yl[i+1]*(qs[i/2]&0x0F00);
    acc[2]+=yl[i+8]*(qs[i/2]&0x00F0);acc[3]+=yl[i+9]*(qs[i/2]&0xF000);
   }
   sumf[row]+=float(as_type<half>(b[0]))*(sumy*-8.f+acc[0]+acc[1]+acc[2]+acc[3]);
  }
  yb+=32*16;
 }
 for(uint row=0;row<NR;++row)out[row]=simd_sum(sumf[row]);
}
template<uint NR>
inline void mv_f16(device const uchar*w,device const float*x,constant uint*p,uint row0,ushort lane,thread float*out){
 device const half4*rows[NR];
 for(uint r=0;r<NR;++r)rows[r]=reinterpret_cast<device const half4*>(w+ulong(min(row0+r,p[1]-1))*p[2]);
 float sumf[NR]={0};
 device const float4*x4=reinterpret_cast<device const float4*>(x);
 for(uint i=lane;i<p[0]/4;i+=32){
  float4 v=x4[i];
  for(uint row=0;row<NR;++row)sumf[row]+=dot(float4(rows[row][i]),v);
 }
 for(uint row=0;row<NR;++row)out[row]=simd_sum(sumf[row]);
}
template<uint NR>
inline void mv_f32(device const uchar*w,device const float*x,constant uint*p,uint row0,ushort lane,thread float*out){
 device const float4*rows[NR];
 for(uint r=0;r<NR;++r)rows[r]=reinterpret_cast<device const float4*>(w+ulong(min(row0+r,p[1]-1))*p[2]);
 float sumf[NR]={0};
 device const float4*x4=reinterpret_cast<device const float4*>(x);
 for(uint i=lane;i<p[0]/4;i+=32){
  float4 v=x4[i];
  for(uint row=0;row<NR;++row)sumf[row]+=dot(rows[row][i],v);
 }
 for(uint row=0;row<NR;++row)out[row]=simd_sum(sumf[row]);
}
// Dense: p = cols, rows, row bytes, kind, expert; grid y is the token.
// Routed: p = cols, rows, row bytes, kind, 0, per-pair input, top-k; grid y is the pair.
// Gate/up variants take both weights and store silu(gate) * up; dense ones
// store gelu(gate) * up (gelu_mul) when p[5].
inline float glu(float g,float u,bool gelu){return gelu?0.5f*g*(1.0f+precise::tanh(0.79788456080286535587989211986876f*g*(1.0f+0.044715f*g*g)))*u:(g/(1.0f+exp(-g)))*u;}
#define MV_ARGS uint2 group [[threadgroup_position_in_grid]],ushort lane [[thread_index_in_simdgroup]],ushort sg [[simdgroup_index_in_threadgroup]],ushort nsg [[simdgroups_per_threadgroup]]
#define MV_STORE(nr,value) for(uint r=0;r<nr;++r)if(!lane&&row0+r<p[1])y[row0+r]=value;
#define GENERIC_MV(name,impl,nr) \
kernel void name(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],MV_ARGS) \
 {uint row0=(group.x*nsg+sg)*nr;float s[nr];impl<nr>(w+ulong(p[4])*p[1]*p[2],x+ulong(group.y)*p[0],p,row0,lane,s);y+=ulong(group.y)*p[1];MV_STORE(nr,s[r])} \
kernel void routed_##name(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device const uint*ids [[buffer(2)]],device float*y [[buffer(3)]],constant uint*p [[buffer(4)]],MV_ARGS) \
 {uint row0=(group.x*nsg+sg)*nr,pair=group.y,input=p[5]?pair:pair/p[6];float s[nr]; \
  impl<nr>(w+ulong(ids[pair])*p[1]*p[2],x+ulong(input)*p[0],p,row0,lane,s);y+=ulong(pair)*p[1];MV_STORE(nr,s[r])} \
kernel void gate_up_##name(device const uchar*wg [[buffer(0)]],device const uchar*wu [[buffer(1)]],device const float*x [[buffer(2)]],device float*y [[buffer(3)]],constant uint*p [[buffer(4)]],MV_ARGS) \
 {uint row0=(group.x*nsg+sg)*nr;float g[nr],u[nr];x+=ulong(group.y)*p[0]; \
  impl<nr>(wg,x,p,row0,lane,g);impl<nr>(wu,x,p,row0,lane,u);y+=ulong(group.y)*p[1];MV_STORE(nr,glu(g[r],u[r],p[5]))} \
kernel void routed_gate_up_##name(device const uchar*wg [[buffer(0)]],device const uchar*wu [[buffer(1)]],device const float*x [[buffer(2)]],device const uint*ids [[buffer(3)]],device float*y [[buffer(4)]],constant uint*p [[buffer(5)]],MV_ARGS) \
 {uint row0=(group.x*nsg+sg)*nr,pair=group.y;ulong e=ulong(ids[pair])*p[1]*p[2];float g[nr],u[nr];x+=ulong(pair/p[6])*p[0]; \
  impl<nr>(wg+e,x,p,row0,lane,g);impl<nr>(wu+e,x,p,row0,lane,u);y+=ulong(pair)*p[1];MV_STORE(nr,glu(g[r],u[r],false))}
GENERIC_MV(mv_q4k_2,mv_q4k,2) GENERIC_MV(mv_q6k_2,mv_q6k,2) GENERIC_MV(mv_q4_0_4,mv_q4_0,4) GENERIC_MV(mv_f16_4,mv_f16,4) GENERIC_MV(mv_f32_4,mv_f32,4)

#define GEMM_TOKENS 64
// Prefill GEMM: 64 rows x GEMM_TOKENS tokens per threadgroup, the tiled_projection
// layout (kernels/qwen.metal, after ggml kernel_mul_mm) with vectorized
// Q4_K/Q6_K/Q4_0 decode of 16 consecutive weights (first % 16 == 0).
template<uint K>
inline float4x4 generic_weights16(device const uchar*row,uint first){
 if(K==12)return mix_weights16(row,first,12);
 float4x4 r;
 if(K==14){
  device const uchar*b=row+(first/256)*210;uint k=first%256,ip=k/128,qq=(k%128)/32,l0=k%32;
  device const uchar*ql=b+64*ip+32*(qq%2)+l0,*qh=b+128+32*ip+l0;
  float d=half_at(b+208)*float(as_type<char>(b[192+8*ip+l0/16+2*qq]));
  for(uint i=0;i<4;++i){
   uint bits=((packed4(ql+4*i)>>(qq>=2?4:0))&0x0f0f0f0fu)|(((packed4(qh+4*i)>>(2*qq))&0x03030303u)<<4);
   r[i]=d*(float4(as_type<uchar4>(bits))-32.0f);
  }
 } else {
  device const uchar*b=row+(first/32)*18;uint h=(first%32)/16;float d=half_at(b);
  for(uint i=0;i<4;++i)r[i]=d*(float4(as_type<uchar4>((packed4(b+2+4*i)>>(4*h))&0x0f0f0f0fu))-8.0f);
 }
 return r;
}
// p: cols, rows, row bytes, kind, expert, tokens. Grid: tokens/NT x rows/64 (token
// tiles of one weight tile run together), 128 threads; SIMD group g computes rows
// 32*(g%2).. of tokens NT/2*(g/2)...
template<uint K,uint NT>
inline void generic_gemm(device const uchar*w,device const float*x,device float*y,constant uint*p,uint2 group,uint tid,uint simd,threadgroup float*scratch){
 constexpr uint MB=NT/16;
 threadgroup half*sa=reinterpret_cast<threadgroup half*>(scratch),*sb=sa+64*32;
 uint cols=p[0],rows=p[1],row0=group.y*64,offset=group.x*NT;
 uint nr=min(64u,rows-row0),nt=min(NT,p[5]-offset),lr=min(tid/2,nr-1),half_k=tid%2;
 device const uchar*wr=w+(ulong(p[4])*rows+row0+lr)*p[2];
 simdgroup_half8x8 ma[4],mb[MB];simdgroup_float8x8 acc[4*MB];
 for(uint i=0;i<4*MB;++i)acc[i]=make_filled_simdgroup_matrix<float,8>(0.0f);
 for(uint base=0;base<cols;base+=32){
  float4x4 decoded=generic_weights16<K>(wr,base+16*half_k);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma unroll
  for(uint i=0;i<16;++i){
   uint sx=2*half_k+i/8,sy=(tid/2)/8,lx=(tid/2)%8,ly=i%8;
   *(sa+64*(8*sx+sy)+8*ly+lx)=decoded[i/4][i%4];
  }
  #pragma unroll
  for(uint item=tid;item<NT*4;item+=128){
   uint token=item/4,k0=8*(item%4),ib=NT/8*(item%4)+token/8;
   *reinterpret_cast<threadgroup half2x4*>(sb+64*ib+8*(token%8))=token<nt?
    half2x4(*reinterpret_cast<device const float2x4*>(x+ulong(offset+token)*cols+base+k0)):half2x4(0.0h);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup const half*a=sa+256*(simd%2),*b=sb+64*MB*(simd/2);
  #pragma unroll
  for(uint k=0;k<4;++k){
   #pragma unroll
   for(uint i=0;i<4;++i)simdgroup_load(ma[i],a+64*i,8,0,false);
   #pragma unroll
   for(uint i=0;i<MB;++i)simdgroup_load(mb[i],b+64*i,8,0,false);
   #pragma unroll
   for(uint i=0;i<4*MB;++i)simdgroup_multiply_accumulate(acc[i],mb[i/4],ma[i%4],acc[i]);
   a+=512;b+=NT*8;
  }
 }
 if(nr==64&&nt==NT){
  device float*dest=y+ulong(offset+8*MB*(simd/2))*rows+row0+32*(simd%2);
  for(uint i=0;i<4*MB;++i)simdgroup_store(acc[i],dest+8*(i%4)+ulong(8*(i/4))*rows,rows,0,false);
 } else {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup float*dest=scratch+32*(simd%2)+8*MB*(simd/2)*64;
  for(uint i=0;i<4*MB;++i)simdgroup_store(acc[i],dest+8*(i%4)+8*64*(i/4),64,0,false);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint i=tid;i<nt*nr;i+=128)y[ulong(offset+i/nr)*rows+row0+i%nr]=scratch[(i/nr)*64+i%nr];
 }
}
#define GENERIC_GEMM(name,k) kernel void name(device const uchar*w [[buffer(0)]],device const float*x [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]){threadgroup float scratch[64*GEMM_TOKENS];generic_gemm<k,GEMM_TOKENS>(w,x,y,p,group,tid,simd,scratch);}
GENERIC_GEMM(gemm_q4k,12) GENERIC_GEMM(gemm_q6k,14) GENERIC_GEMM(gemm_q4_0,2)

// Residual add and RMS norm per row: s = a + b (when p[2]), y = s * rsqrt(mean(s^2) + eps) * w.
// p: width (multiple of 4), eps bits, add. Grid: rows, 256 threads.
kernel void add_rms(device const float*a [[buffer(0)]],device const float*b [[buffer(1)]],device const float*w [[buffer(2)]],device float*s [[buffer(3)]],device float*y [[buffer(4)]],constant uint*p [[buffer(5)]],uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]){
 uint n4=p[0]/4;ulong base=ulong(row)*n4;
 device const float4*a4=reinterpret_cast<device const float4*>(a)+base,*b4=reinterpret_cast<device const float4*>(b)+base,*w4=reinterpret_cast<device const float4*>(w);
 device float4*s4=reinterpret_cast<device float4*>(s)+base,*y4=reinterpret_cast<device float4*>(y)+base;
 threadgroup float partial[8];
 float sum=0;
 for(uint j=tid;j<n4;j+=256){float4 v=p[2]?a4[j]+b4[j]:a4[j];if(p[2])s4[j]=v;sum+=dot(v,v);}
 sum=simd_sum(sum);if(!lane)partial[sg]=sum;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float total=0;for(uint i=0;i<8;++i)total+=partial[i];
 float inv=rsqrt(total/float(p[0])+as_type<float>(p[1]));
 for(uint j=tid;j<n4;j+=256){float4 v=p[2]?a4[j]+b4[j]:a4[j];y4[j]=v*inv*w4[j];}
}

// Per head row (rows of q, then k, then v): optional QK RMS norm, RoPE as in
// rope_generic, q times a scale, then q to qo and k, v into the caches at row
// position + t. p: head dim (<= 512), heads, kv heads, rotated dims, position,
// base bits, NeoX, use factors, QK norm, eps bits, unweighted value norm, q
// scale bits. Grid: heads + 2 kv heads x tokens, 32 threads.
kernel void qk_rope_cache(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device const float*qn [[buffer(3)]],device const float*kn [[buffer(4)]],device const float*factors [[buffer(5)]],device float*qo [[buffer(6)]],device float*kc [[buffer(7)]],device float*vc [[buffer(8)]],constant uint*p [[buffer(9)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
 uint D=p[0],H=p[1],KV=p[2],r=group.x,t=group.y,rot=p[3];
 if(r>=H+KV){
  uint h=r-H-KV;device const float*src=v+(ulong(t)*KV+h)*D;device float*dst=vc+(ulong(p[4]+t)*KV+h)*D;
  float inv=1;
  if(p[10]){float s=0;for(uint j=lane;j<D;j+=32){float x=src[j];s+=x*x;}s=simd_sum(s);inv=rsqrt(s/float(D)+as_type<float>(p[9]));}
  for(uint j=lane;j<D;j+=32)dst[j]=p[10]?src[j]*inv:src[j];
  return;
 }
 threadgroup float row[512];
 bool isq=r<H;uint h=isq?r:r-H;
 device const float*src=isq?q+(ulong(t)*H+h)*D:k+(ulong(t)*KV+h)*D;
 device float*dst=isq?qo+(ulong(t)*H+h)*D:kc+(ulong(p[4]+t)*KV+h)*D;
 device const float*w=isq?qn:kn;
 float inv=1;
 if(p[8]){float s=0;for(uint j=lane;j<D;j+=32){float x=src[j];s+=x*x;}s=simd_sum(s);inv=rsqrt(s/float(D)+as_type<float>(p[9]));}
 for(uint j=lane;j<D;j+=32)row[j]=p[8]?src[j]*inv*w[j]:src[j];
 simdgroup_barrier(mem_flags::mem_threadgroup);
 bool neox=p[6]!=0;float qs=isq?as_type<float>(p[11]):1.0f;
 for(uint j=lane;j<D;j+=32){
  if(j>=rot){dst[j]=row[j]*qs;continue;}
  uint f=neox?j%(rot/2):j/2;uint a=neox?f:(j&~1u);uint b=neox?a+rot/2:a+1;
  float theta=float(p[4]+t)*pow(as_type<float>(p[5]),-2.0f*float(f)/float(rot));if(p[7])theta/=factors[f];
  float c=cos(theta),s=sin(theta);bool first=neox?j<rot/2:(j&1u)==0;
  float y=first?row[a]*c-row[b]*s:row[a]*s+row[b]*c;
  dst[j]=y*qs;
 }
}

// Causal attention for 1-4 query tokens: one threadgroup per (head, token);
// its eight SIMD groups take every eighth key with an online softmax and merge.
// p: head dim, heads, kv heads, position of the first token, sliding window
// (0: none; as flash_prefill). Grid: heads x tokens.
template<uint D>
inline void attention_decode(device const float*q,device const float*k,device const float*v,device float*out,constant uint*p,uint2 group,uint lane,uint sg,threadgroup float*part){
 constexpr uint J=D/32;
 uint h=group.x,t=group.y,kh=h/(p[1]/p[2]),stride=p[2]*D,length=p[3]+t+1,start=p[4]&&length>p[4]?length-p[4]:0;
 const float scale=rsqrt(float(D));
 float qv[J],o[J];
 for(uint j=0;j<J;++j){qv[j]=q[(ulong(t)*p[1]+h)*D+lane+32*j];o[j]=0;}
 float m=-INFINITY,l=0;
 for(uint pos=start+sg;pos<length;pos+=8){
  device const float*kr=k+ulong(pos)*stride+kh*D,*vr=v+ulong(pos)*stride+kh*D;
  float s=0;
  for(uint j=0;j<J;++j)s+=qv[j]*kr[lane+32*j];
  s=simd_sum(s)*scale;
  float next=max(m,s),f=exp(m-next),e=exp(s-next);
  l=l*f+e;m=next;
  for(uint j=0;j<J;++j)o[j]=o[j]*f+e*vr[lane+32*j];
 }
 threadgroup float*ms=part+8*D,*ls=ms+8;
 if(!lane){ms[sg]=m;ls[sg]=l;}
 for(uint j=0;j<J;++j)part[sg*D+lane+32*j]=o[j];
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(sg)return;
 float M=-INFINITY;for(uint i=0;i<8;++i)M=max(M,ms[i]);
 float L=0,w[8];for(uint i=0;i<8;++i){w[i]=exp(ms[i]-M);L+=ls[i]*w[i];}
 for(uint j=0;j<J;++j){
  float acc=0;for(uint i=0;i<8;++i)acc+=part[i*D+lane+32*j]*w[i];
  out[(ulong(t)*p[1]+h)*D+lane+32*j]=acc/L;
 }
}
#define ATTENTION_DECODE(d) kernel void attention_decode_##d(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device const float*v [[buffer(2)]],device float*out [[buffer(3)]],constant uint*p [[buffer(4)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]]){threadgroup float part[8*d+16];attention_decode<d>(q,k,v,out,p,group,lane,sg,part);}
ATTENTION_DECODE(64) ATTENTION_DECODE(96) ATTENTION_DECODE(128) ATTENTION_DECODE(256) ATTENTION_DECODE(512)

// Top-k softmax routing, one SIMD group per token (at most 512 experts): the
// route_gpu_batch selection order and weights with the logits in registers.
// p: experts, k. Grid: tokens, 32 threads.
kernel void route_topk(device const float*logits [[buffer(0)]],device uint*ids [[buffer(1)]],device float*weights [[buffer(2)]],constant uint*p [[buffer(3)]],uint t [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
 uint E=p[0],K=p[1];device const float*l=logits+ulong(t)*E;
 float v[16];
 for(uint i=0;i<16;++i){uint e=lane+32*i;v[i]=e<E?l[e]:-INFINITY;}
 float top=0,sum=0;
 for(uint k=0;k<K;++k){
  float best=-INFINITY;uint index=0xffffffffu;
  for(uint i=0;i<16;++i)if(v[i]>best){best=v[i];index=lane+32*i;}
  float score=simd_max(best);uint e=simd_min(best==score?index:0xffffffffu);
  for(uint i=0;i<16;++i)if(lane+32*i==e)v[i]=-INFINITY;
  if(!k)top=score;
  float w=exp(score-top);sum+=w;
  if(!lane){ids[t*K+k]=e;weights[t*K+k]=w;}
 }
 if(!lane)for(uint k=0;k<K;++k)weights[t*K+k]/=sum;
}
