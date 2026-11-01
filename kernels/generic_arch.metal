// Architecture-specific helpers for the generic path (Gemma 3/4, Phi-3).
// y = x * scale, p: count, scale bits.
kernel void scale_rows(device const float*x [[buffer(0)]],device float*y [[buffer(1)]],constant uint*p [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<p[0])y[i]=x[i]*as_type<float>(p[1]);}
// Final logit soft-capping y = c * tanh(x / c), p: count, c bits.
kernel void softcap_rows(device const float*x [[buffer(0)]],device float*y [[buffer(1)]],constant uint*p [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<p[0]){float c=as_type<float>(p[1]);y[i]=c*precise::tanh(x[i]/c);}}
// GeGLU with the tanh GeLU approximation (ggml_geglu): y = gelu(gate) * up, p: count.
kernel void gelu_mul(device const float*g [[buffer(0)]],device const float*u [[buffer(1)]],device float*y [[buffer(2)]],constant uint*p [[buffer(3)]],uint i [[thread_position_in_grid]]){
 if(i>=p[0])return;float x=g[i];
 y[i]=0.5f*x*(1.0f+precise::tanh(0.79788456080286535587989211986876f*x*(1.0f+0.044715f*x*x)))*u[i];
}
// SwiGLU over fused gate|up rows of width 2*p[1]: y = silu(gate) * up, p: count, ff.
kernel void swiglu_fused(device const float*x [[buffer(0)]],device float*y [[buffer(1)]],constant uint*p [[buffer(2)]],uint i [[thread_position_in_grid]]){
 if(i>=p[0])return;uint t=i/p[1],j=i%p[1];float g=x[ulong(t)*2*p[1]+j];
 y[i]=g/(1.0f+exp(-g))*x[ulong(t)*2*p[1]+p[1]+j];
}
// Copy columns [p[2], p[2]+p[1]) of rows of width p[3]: p: count, width, offset, stride.
kernel void copy_columns(device const float*x [[buffer(0)]],device float*y [[buffer(1)]],constant uint*p [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<p[0])y[i]=x[ulong(i/p[1])*p[3]+p[2]+i%p[1]];}
// Causal attention scores with a sliding window of p[5] keys (llama.cpp
// LLAMA_SWA_TYPE_STANDARD: key j is visible to query q when q - j < window).
// p: head dim, heads, kv heads, length, position of the first row, window.
kernel void attention_scores_window(device const float*q [[buffer(0)]],device const float*k [[buffer(1)]],device float*s [[buffer(2)]],constant uint*p [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]){
 uint d=p[0],t=g.y/p[1],h=g.y%p[1],kh=h/(p[1]/p[2]),j=g.x,qp=p[4]+t;
 if(j>qp||qp-j>=p[5]){if(!lane)s[g.y*p[3]+j]=-INFINITY;return;}
 float v=0;for(uint i=lane;i<d;i+=32)v+=q[g.y*d+i]*k[(ulong(j)*p[2]+kh)*d+i];v=simd_sum(v);if(!lane)s[g.y*p[3]+j]=v*rsqrt(float(d));
}
