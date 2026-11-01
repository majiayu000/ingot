// GLM5-Next F32 operators. Formula references and licenses are recorded in NOTICE.
kernel void glm_add(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                    device float* y [[buffer(2)]], constant uint* p [[buffer(3)]],
                    uint i [[thread_position_in_grid]]) {
    if (i < p[0]) y[i] = a[i] + b[i];
}
// mHC gates are [pre(4), post(4), combination(src,dst)].
kernel void glm_hc_gates(device const float* mix [[buffer(0)]],
                         device const float* base [[buffer(1)]],
                         device const float* scale [[buffer(2)]],
                         device float* gates [[buffer(3)]],
                         constant uint* p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i) return;
    const uint hc = p[0]; const float eps = as_type<float>(p[2]);
    for (uint h = 0; h < hc; ++h) {
        gates[h] = sigmoid_f(mix[h] * scale[0] + base[h]) + eps;
        gates[hc+h] = 2.0f * sigmoid_f(mix[hc+h] * scale[1] + base[hc+h]);
    }
    float comb[16];
    for (uint src = 0; src < hc; ++src) {
        float m = -INFINITY, sum = 0;
        for (uint dst = 0; dst < hc; ++dst)
            m = max(m, mix[2*hc+src*hc+dst]*scale[2]+base[2*hc+src*hc+dst]);
        for (uint dst = 0; dst < hc; ++dst) {
            comb[src*hc+dst] = exp(mix[2*hc+src*hc+dst]*scale[2]+base[2*hc+src*hc+dst]-m);
            sum += comb[src*hc+dst];
        }
        for (uint dst = 0; dst < hc; ++dst) comb[src*hc+dst] = comb[src*hc+dst]/sum + eps;
    }
    for (uint it = 0; it < p[1]; ++it) {
        if (it) for (uint src = 0; src < hc; ++src) {
            float sum = eps;
            for (uint dst = 0; dst < hc; ++dst) sum += comb[src*hc+dst];
            for (uint dst = 0; dst < hc; ++dst) comb[src*hc+dst] /= sum;
        }
        for (uint dst = 0; dst < hc; ++dst) {
            float sum = eps;
            for (uint src = 0; src < hc; ++src) sum += comb[src*hc+dst];
            for (uint src = 0; src < hc; ++src) comb[src*hc+dst] /= sum;
        }
    }
    for (uint j = 0; j < hc*hc; ++j) gates[2*hc+j] = comb[j];
}
kernel void glm_hc_pre(device const float* x [[buffer(0)]], device const float* g [[buffer(1)]],
                       device float* y [[buffer(2)]], constant uint* p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0]) return;
    float sum = 0; for (uint h = 0; h < p[1]; ++h) sum += x[h*p[0]+i]*g[h]; y[i] = sum;
}
kernel void glm_hc_post(device const float* residual [[buffer(0)]], device const float* x [[buffer(1)]],
                        device const float* g [[buffer(2)]], device float* y [[buffer(3)]],
                        constant uint* p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0]*p[1]) return;
    uint dst = i/p[0], j = i%p[0]; float sum = x[j]*g[p[1]+dst];
    for (uint src = 0; src < p[1]; ++src) sum += residual[src*p[0]+j]*g[2*p[1]+src*p[1]+dst];
    y[i] = sum;
}
kernel void glm_hc_mean(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                        constant uint* p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0]) return; float sum = 0;
    for (uint h = 0; h < p[1]; ++h) sum += x[h*p[0]+i]; y[i] = sum/float(p[1]);
}
// KDA decays every key coordinate independently, before the delta update.
kernel void glm_kda(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
                    device const float* v [[buffer(2)]], device const float* f [[buffer(3)]],
                    device const float* beta [[buffer(4)]], device const float* dt [[buffer(5)]],
                    device const float* A [[buffer(6)]], device float* state [[buffer(7)]],
                    device float* y [[buffer(8)]], constant uint* p [[buffer(9)]],
                    uint2 g [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint d = p[0], h = g.y, value = g.x; device float* row = state+(ulong(h)*d+value)*d;
    float pred = 0;
    for (uint j = lane; j < d; j += 32) {
        float z = f[h*d+j]+dt[h*d+j];
        float log_decay = as_type<float>(p[1])*sigmoid_f(-A[h]*z);
        row[j] *= exp(log_decay); pred += row[j]*k[h*d+j];
    }
    pred = simd_sum(pred); float delta = (v[h*d+value]-pred)*sigmoid_f(beta[h]); float out = 0;
    for (uint j = lane; j < d; j += 32) { row[j] += k[h*d+j]*delta; out += row[j]*q[h*d+j]; }
    out = simd_sum(out); if (!lane) y[h*d+value] = out*rsqrt(float(d));
}
// Selection uses biased sigmoid scores; mixture weights remain unbiased.
kernel void glm_route(device const float* logits [[buffer(0)]], device const float* bias [[buffer(1)]],
                      device uint* ids [[buffer(2)]], device float* weights [[buffer(3)]],
                      constant uint* p [[buffer(4)]], uint lane [[thread_index_in_simdgroup]]) {
    float sum = 0;
    for (uint t = 0; t < p[1]; ++t) {
        uint index = 0; float local = -INFINITY;
        for (uint e = lane; e < p[0]; e += 32) {
            bool used = false; for (uint j = 0; j < t; ++j) used |= ids[j] == e;
            float candidate = sigmoid_f(logits[e])+bias[e];
            if (!used && candidate > local) { local = candidate; index = e; }
        }
        float score = simd_max(local);
        uint best = simd_min(local == score ? index : 0xffffffffu);
        if (!lane) { ids[t] = best; weights[t] = sigmoid_f(logits[best]); sum += weights[t]; }
        simdgroup_barrier(mem_flags::mem_device);
    }
    if (!lane) for (uint t = 0; t < p[1]; ++t) weights[t] = weights[t]/sum*as_type<float>(p[2]);
}

kernel void glm_swiglu(device const float* gate [[buffer(0)]], device const float* up [[buffer(1)]],
                       device float* y [[buffer(2)]], constant uint* p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0]) return; float limit = as_type<float>(p[1]);
    float g = min(gate[i], limit), u = clamp(up[i], -limit, limit); y[i] = g*sigmoid_f(g)*u;
}
kernel void glm_merge(device const float* routed [[buffer(0)]], device const float* shared [[buffer(1)]],
                      device const float* weights [[buffer(2)]], device float* y [[buffer(3)]],
                      constant uint* p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0]) return; float sum = 0;
    for (uint e = 0; e < p[1]; ++e) sum += routed[e*p[0]+i]*weights[e]; y[i] = sum+shared[i];
}
kernel void glm_layer_norm(device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                           device const float* bias [[buffer(2)]], device float* y [[buffer(3)]],
                           constant uint* p [[buffer(4)]], uint lane [[thread_index_in_simdgroup]]) {
    float sum = 0; for (uint j = lane; j < p[0]; j += 32) sum += x[j];
    float mean = simd_sum(sum)/float(p[0]); float var = 0;
    for (uint j = lane; j < p[0]; j += 32) { float z = x[j]-mean; var += z*z; }
    float inv = rsqrt(simd_sum(var)/float(p[0])+as_type<float>(p[1]));
    for (uint j = lane; j < p[0]; j += 32) y[j] = (x[j]-mean)*inv*w[j]+bias[j];
}
// Only completed four-token pools enter selection. The unfinished tail stays visible.
kernel void glm_pool(device const float* keys [[buffer(0)]], device const float* gates [[buffer(1)]],
                     device const float* ape [[buffer(2)]], device float* pooled [[buffer(3)]],
                     constant uint* p [[buffer(4)]], uint j [[thread_position_in_grid]]) {
    if (j >= p[0]) return; uint start = p[2]*p[1]; float m = -INFINITY;
    for (uint t = 0; t < p[1]; ++t) m = max(m,gates[(start+t)*p[0]+j]+ape[t*p[0]+j]);
    float sum = 0, value = 0;
    for (uint t = 0; t < p[1]; ++t) { float a = exp(gates[(start+t)*p[0]+j]+ape[t*p[0]+j]-m); sum += a; value += a*keys[(start+t)*p[0]+j]; }
    pooled[p[2]*p[0]+j] = value/sum;
}
kernel void glm_index_score(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
                            device const float* weights [[buffer(2)]], device float* score [[buffer(3)]],
                            constant uint* p [[buffer(4)]], uint b [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    float total = 0;
    for (uint h = 0; h < p[1]; ++h) {
        float dot = 0; for (uint j = lane; j < p[0]; j += 32) dot += q[h*p[0]+j]*k[b*p[0]+j];
        dot = simd_sum(dot); total += max(dot,0.0f)*weights[h]*rsqrt(float(p[0]*p[1]));
    }
    if (!lane) score[b] = total;
}
// One rank calculation per completed pool, not once per token in that pool.
kernel void glm_index_mask(device const float* score [[buffer(0)]], device float* mask [[buffer(1)]],
                           constant uint* p [[buffer(2)]], uint block [[thread_position_in_grid]]) {
    uint complete=p[0]/p[1];
    if(!block)for(uint t=complete*p[1];t<p[0];++t)mask[t]=1;
    if(block>=complete)return;
    uint rank=0;
    if(complete>p[2]/p[1])for(uint b=0;b<complete;++b)rank+=score[b]>score[block]||(score[b]==score[block]&&b<block);
    float visible=(complete<=p[2]/p[1]||rank<p[2]/p[1])?1.0f:0.0f;
    for(uint t=0;t<p[1];++t)mask[block*p[1]+t]=visible;
}
kernel void glm_mla_scores(device const float* q [[buffer(0)]], device const float* latent [[buffer(1)]],
                           device const float* mask [[buffer(2)]], device float* score [[buffer(3)]],
                           constant uint* p [[buffer(4)]], uint2 g [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    if(mask[g.x]<=0){if(!lane)score[g.y*p[1]+g.x]=-INFINITY;return;}
    float dot = 0; for (uint j = lane; j < p[0]; j += 32) dot += q[g.y*p[0]+j]*latent[ulong(g.x)*p[0]+j];
    dot = simd_sum(dot); if (!lane) score[g.y*p[1]+g.x] = mask[g.x]>0 ? dot*rsqrt(float(p[2])) : -INFINITY;
}

// GLM selects whole completed pools plus the unfinished tail. The Qwen mask
// uses a different token budget; this model-specific rule must remain separate.
kernel void glm_index_mask_sorted(device const float*score [[buffer(0)]],
                                  device float*mask [[buffer(1)]],
                                  constant uint*p [[buffer(2)]],
                                  uint tid [[thread_index_in_threadgroup]]) {
 threadgroup float values[4096];threadgroup uint ids[4096];
 uint blocks=p[0]/p[1],n=1;while(n<blocks)n*=2;
 for(uint i=tid;i<n;i+=32){values[i]=i<blocks?score[i]:-INFINITY;ids[i]=i<blocks?i:0xffffffffu;}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint size=2;size<=n;size*=2)for(uint stride=size/2;stride;stride/=2){
  for(uint i=tid;i<n;i+=32){uint j=i^stride;if(j>i){float a=values[i],b=values[j];uint ai=ids[i],bi=ids[j];bool before=a>b||(a==b&&ai<bi);if(before==((i&size)!=0)){values[i]=b;values[j]=a;ids[i]=bi;ids[j]=ai;}}}
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 if(!tid)for(uint t=blocks*p[1];t<p[0];++t)mask[t]=1;
 for(uint rank=tid;rank<blocks;rank+=32){float keep=rank<p[2]/p[1]?1.0f:0.0f;for(uint t=0;t<p[1];++t)mask[ids[rank]*p[1]+t]=keep;}
}
// Absorbed MLA query width is latent, but the model's attention scale uses qk.
// Indices keep each original position's modulo-32 reduction lane.
kernel void glm_mla_scores_compact(device const float*q [[buffer(0)]],
                                  device const float*latent [[buffer(1)]],
                                  device const uint*ids [[buffer(2)]],
                                  device const uint*offsets [[buffer(3)]],
                                  device float*score [[buffer(4)]],
                                  constant uint*p [[buffer(5)]],
                                  uint2 g [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]]) {
 if(g.x>=offsets[32])return;
 uint pos=ids[g.x];float dot=0;
 for(uint j=lane;j<p[0];j+=32)dot+=q[g.y*p[0]+j]*latent[ulong(pos)*p[0]+j];
 dot=simd_sum(dot);if(!lane)score[g.y*p[1]+g.x]=dot*rsqrt(float(p[2]));
}

// The verified GLM binding has a 512-wide absorbed query/value and qk=256.
// Ascending selected positions give dense and sparse paths the same arithmetic.
// A part holds a stable max, denominator and unnormalized value sum for 64 keys.
kernel void glm_mla_online_parts(device const float*q [[buffer(0)]],
 device const float*latent [[buffer(1)]],device const uint*ids [[buffer(2)]],
 device const uint*offsets [[buffer(3)]],device float*partial [[buffer(4)]],
 constant uint*p [[buffer(5)]],uint2 g [[threadgroup_position_in_grid]],
 uint lane [[thread_index_in_simdgroup]]) {
 uint length=p[2]?offsets[32]:p[0],begin=g.x*64,end=min(begin+64,length);
 uint base=(g.y*p[1]+g.x)*514;
 float maximum=-INFINITY,denominator=0,values[16];
 for(uint j=0;j<16;++j)values[j]=0;
 for(uint i=begin;i<end;++i){
  uint pos=p[2]?ids[i]:i;float score=0;
  for(uint j=lane;j<512;j+=32)score+=q[g.y*512+j]*latent[ulong(pos)*512+j];
  score=simd_sum(score)*0.0625f;
  float next=max(maximum,score),old_weight=exp(maximum-next),new_weight=exp(score-next);
  denominator=denominator*old_weight+new_weight;
  for(uint j=0;j<16;++j)values[j]=values[j]*old_weight+new_weight*latent[ulong(pos)*512+lane+j*32];
  maximum=next;
 }
 for(uint j=0;j<16;++j)partial[base+lane+j*32]=values[j];
 if(!lane){partial[base+512]=maximum;partial[base+513]=denominator;}
}
kernel void glm_mla_online_merge(device const float*partial [[buffer(0)]],
 device float*y [[buffer(1)]],constant uint*p [[buffer(2)]],
 uint head [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
 uint parts=p[0];float maximum=-INFINITY,denominator=0,values[16];
 for(uint j=0;j<16;++j)values[j]=0;
 for(uint i=0;i<parts;++i)maximum=max(maximum,partial[(head*parts+i)*514+512]);
 for(uint i=0;i<parts;++i){
  uint base=(head*parts+i)*514;float count=partial[base+513];
  if(count==0)continue;
  float weight=exp(partial[base+512]-maximum);denominator+=count*weight;
  for(uint j=0;j<16;++j)values[j]+=partial[base+lane+j*32]*weight;
 }
 for(uint j=0;j<16;++j)y[head*512+lane+j*32]=denominator>0?values[j]/denominator:0;
}
