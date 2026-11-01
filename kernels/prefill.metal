// Expert grouping for resident packed prefill, on the GPU so no host sync sits
// between routing and the expert GEMMs. Same stable layout as the former CPU
// grouping: experts ascending, pair indices ascending within an expert, 32-token
// tiles, scalar mode for one-token tiles; unused tiles have zero tokens.
// layout: per expert [tokens, first pair]; tiles: [expert, first pair, tokens, mode].
// p: [pairs, experts (<=512), tile capacity]. One 256-thread group.
kernel void expert_layout(device const uint*ids [[buffer(0)]],device uint*layout [[buffer(1)]],device uint*tiles [[buffer(2)]],constant uint*p [[buffer(3)]],uint tid [[thread_index_in_threadgroup]]) {
 threadgroup atomic_uint counts[512];threadgroup uint first[512],tile0[513];
 uint experts=p[1];
 for(uint e=tid;e<experts;e+=256)atomic_store_explicit(&counts[e],0u,memory_order_relaxed);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint i=tid;i<p[0];i+=256)atomic_fetch_add_explicit(&counts[ids[i]],1u,memory_order_relaxed);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(tid==0){uint a=0,b=0;for(uint e=0;e<experts;++e){uint n=atomic_load_explicit(&counts[e],memory_order_relaxed);first[e]=a;tile0[e]=b;a+=n;b+=(n+31)/32;}tile0[experts]=b;}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint e=tid;e<experts;e+=256) {
  uint n=atomic_load_explicit(&counts[e],memory_order_relaxed);layout[2*e]=n;layout[2*e+1]=first[e];
  for(uint k=0;32*k<n;++k){uint nt=min(32u,n-32*k),t=tile0[e]+k;tiles[4*t]=e;tiles[4*t+1]=first[e]+32*k;tiles[4*t+2]=nt;tiles[4*t+3]=nt>=2?1u:0u;}
 }
 for(uint t=tile0[experts]+tid;t<p[2];t+=256){tiles[4*t]=0;tiles[4*t+1]=0;tiles[4*t+2]=0;tiles[4*t+3]=0;}
}

// One 256-thread group per expert writes its pair indices in ascending order,
// and each pair's packed position (inverse) for combine_packed.
kernel void expert_pairs(device const uint*ids [[buffer(0)]],device const uint*layout [[buffer(1)]],device uint*pairs [[buffer(2)]],device uint*inverse [[buffer(3)]],constant uint*p [[buffer(4)]],uint e [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]]) {
 if(layout[2*e]==0)return;
 threadgroup uint before[256];
 uint chunk=(p[0]+255)/256,start=min(p[0],tid*chunk),end=min(p[0],start+chunk),mine=0;
 for(uint i=start;i<end;++i)mine+=ids[i]==e;
 before[tid]=mine;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(tid==0){uint a=0;for(uint k=0;k<256;++k){uint v=before[k];before[k]=a;a+=v;}}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 uint at=layout[2*e+1]+before[tid];
 for(uint i=start;i<end;++i)if(ids[i]==e){inverse[i]=at;pairs[at++]=i;}
}

// combine_batch over packed expert outputs: the same per-token sum in route order.
kernel void combine_packed(device const float*routed [[buffer(0)]],device const float*shared [[buffer(1)]],device const float*weights [[buffer(2)]],device const float*gate [[buffer(3)]],device const uint*inverse [[buffer(4)]],device float*out [[buffer(5)]],constant uint*p [[buffer(6)]],uint i [[thread_position_in_grid]]){if(i>=p[0]*p[2])return;uint t=i/p[0],j=i%p[0];float sum=0;for(uint e=0;e<p[1];++e)sum+=routed[ulong(inverse[t*p[1]+e])*p[0]+j]*weights[t*p[1]+e];out[i]=sum+shared[i]/(1.0f+exp(-gate[t]));}
