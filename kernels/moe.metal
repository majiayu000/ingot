// Quantization layouts adapted from ggml; see kernels/NOTICE.
#include <metal_stdlib>
using namespace metal;
constant uint projection_type [[function_constant(0)]];

inline float half_at(device const uchar *p) {
    // All supported quantization blocks and half offsets are 2-byte aligned.
    return float(*reinterpret_cast<device const half *>(p));
}

inline float weight_at(device const uchar *row, uint type, uint j) {
    if (mix_type(type)) return mix_weight_at(row, type, j);
    if (type == 0) return reinterpret_cast<device const float *>(row)[j];
    if (type == 1) return half_at(row+2*j);
    if (type == 2) { device const uchar *b=row+(j/32)*18; uint k=j%32; return half_at(b)*float(int((b[2+k%16]>>(4*(k/16)))&15)-8); }
    if (type == 8) { device const uchar *b=row+(j/32)*34; return half_at(b)*float(as_type<char>(b[2+j%32])); }
    if (type == 10) { // Q2_K: sixteen 16-element scale/min groups per block.
        device const uchar *b=row+(j/256)*84; uint k=j%256;
        uint sc=b[k/16],q=(b[16+(k/128)*32+k%32]>>(2*((k%128)/32)))&3;
        return half_at(b+80)*float(sc&15)*float(q)-half_at(b+82)*float(sc>>4);
    }
    if (type == 14) {
        device const uchar *b=row+(j/256)*210; uint k=j%256,n=k/128,l=k%32,t=(k%128)/32;
        uint low=(b[n*64+l+(t%2)*32]>>(t>=2?4:0))&15;
        uint high=(b[128+n*32+l]>>(2*t))&3;
        int scale=as_type<char>(b[192+n*8+l/16+2*t]);
        return half_at(b+208)*scale*(int(low|(high<<4))-32);
    }
    if (type == 16) {
        device const uchar *b=row+(j/256)*66;uint k=j%256,g=k/32,l=(k%32)/8,v=k%8;
        device const uchar *a=b+2+g*8;
        uint packed=uint(a[4])|(uint(a[5])<<8)|(uint(a[6])<<16)|(uint(a[7])<<24);
        float grid=float((iq2xxs_grid[a[l]]>>(8*v))&255);
        uint signs=ksigns_iq2xs[(packed>>(7*l))&127];
        return half_at(b)*(0.5f+(packed>>28))*0.25f*grid*((signs&(1<<v))?-1.0f:1.0f);
    }
    if (type == 29) {
        device const uchar *b=row+(j/256)*56;uint k=j%256,g=k/32,l=(k%32)/8,v=k%8;
        uint sc[4]; for(uint i=0;i<4;++i)sc[i]=uint(b[48+2*i])|(uint(b[49+2*i])<<8);
        uint bits=(sc[0]>>12)|((sc[1]>>8)&240)|((sc[2]>>4)&3840)|(sc[3]&61440);
        uint high=b[32+g*2+l/2];uint index=b[g*4+l]|(((high>>(4*(l%2)))&7)<<8);
        float delta=(high&(8<<(4*(l%2))))?-0.125f:0.125f;
        float grid=float(as_type<char>(uchar((iq1s_grid[index]>>(8*v))&255)));
        uint scale=(sc[g/2]>>(6*(g%2)+3*(l/2)))&7;
        return float(as_type<half>(ushort(bits)))*(2*scale+1)*(grid+delta);
    }
    if (type == 30) { // BF16
        uint bits = uint(reinterpret_cast<device const ushort *>(row)[j]) << 16;
        return as_type<float>(bits);
    }
    if (type == 42) { // GSQ Q2_0: consecutive 2-bit codes, 64 elements/block.
        device const uchar *b = row + (j / 64) * 18;
        uint k = j % 64;
        return half_at(b) * (int((b[2+k/4] >> (2*(k%4))) & 3) - 1);
    }
    if (type == 22) { // IQ2_S
        device const uchar *b = row + (j / 256) * 82;
        uint k = j % 256, group = k / 32, l = (k % 32) / 8, v = k % 8;
        uint index = b[2+4*group+l] | ((uint(b[66+group]) << (8-2*l)) & 0x300);
        float grid = float((iq2s_grid[index] >> (8*v)) & 255);
        uint scale = (b[74+group] >> (4*(l/2))) & 15;
        float sign = (b[34+4*group+l] & (1 << v)) ? -1.0f : 1.0f;
        return half_at(b) * (0.5f + scale) * 0.25f * grid * sign;
    }
    if (type == 21) { // IQ3_S
        device const uchar *b = row + (j / 256) * 110;
        uint k = j % 256, group = k / 32, l = (k % 32) / 8, v = k % 8;
        uint half_grid = v / 4;
        uint index = b[2+group*8+2*l+half_grid]
            | ((uint(b[66+group]) << (8-2*l-half_grid)) & 256);
        float grid = float((iq3s_grid[index] >> (8*(v%4))) & 255);
        uint scale = (b[106+group/2] >> (4*(group%2))) & 15;
        float sign = (b[74+group*4+l] & (1 << v)) ? -1.0f : 1.0f;
        return half_at(b) * (1+2*scale) * grid * sign;
    }
    if (type == 20) { // IQ4_NL
        device const uchar *b = row + (j / 32) * 18;
        uint k = j % 32, q = (b[2+k%16] >> (4*(k/16))) & 15;
        return half_at(b) * kvalues_iq4nl[q];
    }
    // IQ4_XS. Rust rejects all unsupported formats before reaching this kernel.
    device const uchar *b = row + (j / 256) * 136;
    uint k = j % 256, group = k / 32, v = k % 32;
    uint high = uint(b[2]) | (uint(b[3]) << 8);
    int scale = int(((b[4+group/2] >> (4*(group%2))) & 15)
        | (((high >> (2*group)) & 3) << 4)) - 32;
    uint q = (b[8+group*16+v%16] >> (4*(v/16))) & 15;
    return half_at(b) * scale * kvalues_iq4nl[q];
}

// Decode a contiguous, aligned group of 16 values. The hot expert formats
// share metadata and codebook loads. F32 expressions match weight_at exactly.
// The grouping follows ggml's dequantize.h; see NOTICE for fixed provenance.
inline float4x4 weights16(device const uchar *row, uint first, uint cols) {
    float4x4 result;
    if (first + 16 > cols) {
        for (uint i=0;i<16;++i) result[i/4][i%4]=first+i<cols?weight_at(row,projection_type,first+i):0.0f;
        return result;
    }
    if(projection_type==8) {
        device const uchar*b=row+(first/32)*34;float d=half_at(b);
        for(uint j=0;j<4;++j){device const uchar*q=b+2+first%32+4*j;result[j]=d*float4(char4(as_type<char>(q[0]),as_type<char>(q[1]),as_type<char>(q[2]),as_type<char>(q[3])));}
    } else if(projection_type==30) {
        device const ushort4*data=reinterpret_cast<device const ushort4*>(row)+first/4;
        for(uint j=0;j<4;++j)result[j]=as_type<float4>(uint4(data[j])<<16);
    } else if(projection_type==21) {
        device const uchar*b=row+(first/256)*110;uint g=(first%256)/32,half_group=(first%32)/16;
        float d=half_at(b)*(1+2*((b[106+g/2]>>(4*(g%2)))&15));
        for(uint j=0;j<4;++j){uint l=2*half_group+j/2,h=j%2;uint index=b[2+g*8+2*l+h]|((uint(b[66+g])<<(8-2*l-h))&256);uint signs=b[74+g*4+l]>>(4*h);
            float4 v=d*float4(as_type<uchar4>(iq3s_grid[index]));result[j]=select(v,-v,(uint4(signs)&uint4(1,2,4,8))!=uint4(0));}
    } else if (projection_type==42) {
        device const uchar *b=row+(first/64)*18;
        float d=half_at(b);
        for(uint i=0;i<4;++i) {
            uint q=b[2+(first%64)/4+i];
            for(uint j=0;j<4;++j) result[i][j]=d*(int((q>>(2*j))&3)-1);
        }
    } else if (projection_type==16 || projection_type==22) {
        device const uchar *b=row+(first/256)*(projection_type==16?66:82);
        uint g=(first%256)/32, half_group=(first%32)/16;
        float d=half_at(b),scale;uint packed=0;
        if(projection_type==16) {
            device const uchar *a=b+2+g*8;
            packed=uint(a[4])|(uint(a[5])<<8)|(uint(a[6])<<16)|(uint(a[7])<<24);
            scale=d*(0.5f+(packed>>28))*0.25f;
        } else scale=d*(0.5f+((b[74+g]>>(4*half_group))&15))*0.25f;
        for(uint h=0;h<2;++h) {
            uint l=2*half_group+h;
            constant uchar4* grid;uint signs;
            if(projection_type==16) {
                grid=reinterpret_cast<constant uchar4*>(iq2xxs_grid+b[2+g*8+l]);
                signs=ksigns_iq2xs[(packed>>(7*l))&127];
            } else {
                uint index=b[2+4*g+l]|((uint(b[66+g])<<(8-2*l))&0x300);
                grid=reinterpret_cast<constant uchar4*>(iq2s_grid+index);signs=b[34+4*g+l];
            }
            float4 low=scale*float4(grid[0]),high=scale*float4(grid[1]);
            result[2*h]=select(low,-low,(uint4(signs)&uint4(1,2,4,8))!=uint4(0));
            result[2*h+1]=select(high,-high,(uint4(signs)&uint4(16,32,64,128))!=uint4(0));
        }
    } else if (projection_type==29) {
        device const uchar *b=row+(first/256)*56;
        uint g=(first%256)/32, half_group=(first%32)/16,sc[4];
        for(uint i=0;i<4;++i)sc[i]=uint(b[48+2*i])|(uint(b[49+2*i])<<8);
        uint bits=(sc[0]>>12)|((sc[1]>>8)&240)|((sc[2]>>4)&3840)|(sc[3]&61440);
        uint scale=(sc[g/2]>>(6*(g%2)+3*half_group))&7;
        float d=float(as_type<half>(ushort(bits)))*(2*scale+1);
        uint high=b[32+g*2+half_group];
        for(uint h=0;h<2;++h) {
            uint index=b[g*4+2*half_group+h]|(((high>>(4*h))&7)<<8);
            constant char4* grid=reinterpret_cast<constant char4*>(iq1s_grid+index);float delta=(high&(8<<(4*h)))?-0.125f:0.125f;
            result[2*h]=d*(float4(grid[0])+delta);result[2*h+1]=d*(float4(grid[1])+delta);
        }
    } else if (projection_type==23 || projection_type==20) {
        device const uchar *b=row+(first/(projection_type==23?256:32))*(projection_type==23?136:18);
        uint g=(first%256)/32,shift=4*((first%32)/16),offset=2;
        float d=half_at(b);
        if(projection_type==23) {
            uint high=uint(b[2])|(uint(b[3])<<8);
            int scale=int(((b[4+g/2]>>(4*(g%2)))&15)|(((high>>(2*g))&3)<<4))-32;
            d*=scale;offset=8+g*16;
        }
        for(uint i=0;i<16;++i)result[i/4][i%4]=d*kvalues_iq4nl[(b[offset+i]>>shift)&15];
    } else if (mix_type(projection_type)) {
        result=mix_weights16(row,first,projection_type);
    } else {
        for(uint i=0;i<16;++i)result[i/4][i%4]=weight_at(row,projection_type,first+i);
    }
    return result;
}

// Block metadata is decoded once per lane while retaining the original lane
// accumulation order (lane, lane+32, ...). No reduced precision or fast math.
// Four independent accumulators reuse each decoded weight and block scale.
// Count 1 callers select .x; the remaining components are dead code.
inline float4 quant_inputs4(device const float*x,uint j,uint stride,uint count) {
    return float4(x[j],count>1?x[j+stride]:0.0f,count>2?x[j+2*stride]:0.0f,count>3?x[j+3*stride]:0.0f);
}
inline float4 dot_quant_rows(device const uchar *row, device const float *x,
                           uint cols, uint lane, uint input_stride, uint count) {
    float4 sum=0.0f;
    if (projection_type==42) {
        for(uint base=0;base<cols;base+=64) {
            device const uchar *b=row+(base/64)*18;
            float d=half_at(b);
            uint q0=(b[2+lane/4]>>(2*(lane%4)))&3;
            uint q1=(b[10+lane/4]>>(2*(lane%4)))&3;
            sum+=(d*(int(q0)-1))*quant_inputs4(x,base+lane,input_stride,count);
            sum+=(d*(int(q1)-1))*quant_inputs4(x,base+lane+32,input_stride,count);
        }
    } else if (projection_type==10) {
        for(uint base=0;base<cols;base+=256) {
            device const uchar *b=row+(base/256)*84;
            float d=half_at(b+80),m=half_at(b+82);
            uint q0=b[16+lane],q1=b[48+lane];
            for(uint g=0;g<8;++g) {
                uint k=lane+32*g,sc=b[k/16],q=((g<4?q0:q1)>>(2*(g%4)))&3;
                float w=d*float(sc&15)*float(q)-m*float(sc>>4);
                sum+=w*quant_inputs4(x,base+k,input_stride,count);
            }
        }
    } else if (projection_type==21) {
        uint l=lane/8,v=lane%8,half_grid=v/4;
        for(uint base=0;base<cols;base+=256) {
            device const uchar*b=row+(base/256)*110;
            float d=half_at(b);
            for(uint g=0;g<8;++g) {
                uint index=b[2+g*8+2*l+half_grid]|((uint(b[66+g])<<(8-2*l-half_grid))&256);
                float grid=float((iq3s_grid[index]>>(8*(v%4)))&255);
                uint scale=(b[106+g/2]>>(4*(g%2)))&15;
                float sign=(b[74+g*4+l]&(1<<v))?-1.0f:1.0f;
                float w=d*(1+2*scale)*grid*sign;
                sum+=w*quant_inputs4(x,base+lane+32*g,input_stride,count);
            }
        }
    } else if (projection_type==22) {
        uint l=lane/8,v=lane%8;
        for(uint base=0;base<cols;base+=256) {
            device const uchar *b=row+(base/256)*82;
            float d=half_at(b);
            for(uint g=0;g<8;++g) {
                uint index=b[2+4*g+l]|((uint(b[66+g])<<(8-2*l))&0x300);
                float grid=float((iq2s_grid[index]>>(8*v))&255);
                uint scale=(b[74+g]>>(4*(l/2)))&15;
                float sign=(b[34+4*g+l]&(1<<v))?-1.0f:1.0f;
                float w=d*(0.5f+scale)*0.25f*grid*sign;
                sum+=w*quant_inputs4(x,base+lane+32*g,input_stride,count);
            }
        }
    } else if (projection_type==16) {
        uint l=lane/8,v=lane%8;
        for(uint base=0;base<cols;base+=256) {
            device const uchar *b=row+(base/256)*66;
            float d=half_at(b);
            for(uint g=0;g<8;++g) {
                device const uchar *a=b+2+g*8;
                uint packed=uint(a[4])|(uint(a[5])<<8)|(uint(a[6])<<16)|(uint(a[7])<<24);
                float grid=float((iq2xxs_grid[a[l]]>>(8*v))&255);
                uint signs=ksigns_iq2xs[(packed>>(7*l))&127];
                float w=d*(0.5f+(packed>>28))*0.25f*grid*((signs&(1<<v))?-1.0f:1.0f);
                sum+=w*quant_inputs4(x,base+lane+32*g,input_stride,count);
            }
        }
    } else if (projection_type==23) {
        for(uint base=0;base<cols;base+=256) {
            device const uchar *b=row+(base/256)*136;
            float d=half_at(b);uint high=uint(b[2])|(uint(b[3])<<8);
            for(uint g=0;g<8;++g) {
                int scale=int(((b[4+g/2]>>(4*(g%2)))&15)|(((high>>(2*g))&3)<<4))-32;
                uint q=(b[8+g*16+lane%16]>>(4*(lane/16)))&15;
                sum+=(d*scale*kvalues_iq4nl[q])*quant_inputs4(x,base+lane+32*g,input_stride,count);
            }
        }
    } else if (projection_type==29) {
        for(uint base=0;base<cols;base+=256) {
            device const uchar *b=row+(base/256)*56;
            uint sc[4];for(uint i=0;i<4;++i)sc[i]=uint(b[48+2*i])|(uint(b[49+2*i])<<8);
            uint bits=(sc[0]>>12)|((sc[1]>>8)&240)|((sc[2]>>4)&3840)|(sc[3]&61440);
            float d=float(as_type<half>(ushort(bits)));
            uint l=lane/8,v=lane%8;
            for(uint g=0;g<8;++g) {
                uint high=b[32+g*2+l/2],index=b[g*4+l]|(((high>>(4*(l%2)))&7)<<8);
                float delta=(high&(8<<(4*(l%2))))?-0.125f:0.125f;
                float grid=float(as_type<char>(uchar((iq1s_grid[index]>>(8*v))&255)));
                uint scale=(sc[g/2]>>(6*(g%2)+3*(l/2)))&7;
                sum+=(d*(2*scale+1)*(grid+delta))*quant_inputs4(x,base+lane+32*g,input_stride,count);
            }
        }
    } else if (projection_type==30 && cols==10240) {
        // Long BF16 HC projections: four weights per lane, F32 accumulation.
        // Short HC-up and router shapes retain their measured faster layout.
        device const ushort4 *v=reinterpret_cast<device const ushort4*>(row);
        for(uint j=4*lane;j<cols;j+=128) {
            float4 w=as_type<float4>(uint4(v[j/4])<<16);
            sum+=w.x*quant_inputs4(x,j,input_stride,count);
            sum+=w.y*quant_inputs4(x,j+1,input_stride,count);
            sum+=w.z*quant_inputs4(x,j+2,input_stride,count);
            sum+=w.w*quant_inputs4(x,j+3,input_stride,count);
        }
    } else if (mix_type(projection_type)) {
        sum=mix_dot_rows(row,x,cols,lane,input_stride,count,projection_type);
    } else {
        for(uint j=lane;j<cols;j+=32)sum+=weight_at(row,projection_type,j)*quant_inputs4(x,j,input_stride,count);
    }
    return sum;
}
// A lane owns a 32-value quant group, sharing its scale and codebook metadata.
// Based on the fixed ggml IQ2 matvec layout in NOTICE; all arithmetic stays F32.
inline float dot_iq2_blocks(device const uchar*row,device const float*x,uint cols,uint lane) {
    float sum=0.0f;
    for(uint at=lane*32;at<cols;at+=32*32) {
        uint group=(at%256)/32;
        device const uchar*b=row+(at/256)*(projection_type==22?82:66);
        float d=half_at(b);
        uint packed=0;
        if(projection_type==16) {
            device const uchar*a=b+2+group*8;
            packed=uint(a[4])|(uint(a[5])<<8)|(uint(a[6])<<16)|(uint(a[7])<<24);
        }
        float sub=0.0f;
        #pragma unroll
        for(uint l=0;l<4;++l) {
            uint index,signs;float scale;
            if(projection_type==22) {
                index=b[2+4*group+l]|((uint(b[66+group])<<(8-2*l))&0x300);
                signs=b[34+4*group+l];
                scale=0.5f+((b[74+group]>>(4*(l/2)))&15);
            } else {
                index=b[2+group*8+l];
                signs=ksigns_iq2xs[(packed>>(7*l))&127];
                scale=0.5f+(packed>>28);
            }
            ulong grid=projection_type==22?iq2s_grid[index]:iq2xxs_grid[index];
            float part=0.0f;
            #pragma unroll
            for(uint j=0;j<8;++j) {
                float weight=float((grid>>(j*8))&255)*((signs&(1<<j))?-1.0f:1.0f);
                part+=x[at+8*l+j]*weight;
            }
            sub+=scale*part;
        }
        sum+=d*sub;
    }
    return sum*0.25f;
}
inline float4 dot_iq2_rows(device const uchar*rows,device const float*x,uint cols,uint stride,uint valid,uint lane) {
    float4 sum=0.0f;
    for(uint at=lane*32;at<cols;at+=32*32) {
        float4 values[8];
        #pragma unroll
        for(uint j=0;j<8;++j)values[j]=*reinterpret_cast<device const float4*>(x+at+4*j);
        #pragma unroll
        for(uint output=0;output<4;++output) {
        if(output>=valid)continue;
        device const uchar*row=rows+ulong(output)*stride;
        uint group=(at%256)/32;
        device const uchar*b=row+(at/256)*(projection_type==22?82:66);
        float d=half_at(b);
        uint packed=0;
        if(projection_type==16) {
            device const uchar*a=b+2+group*8;
            packed=uint(a[4])|(uint(a[5])<<8)|(uint(a[6])<<16)|(uint(a[7])<<24);
        }
        float sub=0.0f;
        #pragma unroll
        for(uint l=0;l<4;++l) {
            uint index,signs;float scale;
            if(projection_type==22) {
                index=b[2+4*group+l]|((uint(b[66+group])<<(8-2*l))&0x300);
                signs=b[34+4*group+l];
                scale=0.5f+((b[74+group]>>(4*(l/2)))&15);
            } else {
                index=b[2+group*8+l];
                signs=ksigns_iq2xs[(packed>>(7*l))&127];
                scale=0.5f+(packed>>28);
            }
            constant uchar4* grid=projection_type==22?reinterpret_cast<constant uchar4*>(iq2s_grid+index):reinterpret_cast<constant uchar4*>(iq2xxs_grid+index);
            float4 lo=float4(grid[0]),hi=float4(grid[1]);
            lo=select(lo,-lo,(uint4(signs)&uint4(1,2,4,8))!=uint4(0));
            hi=select(hi,-hi,(uint4(signs)&uint4(16,32,64,128))!=uint4(0));
            float part=dot(values[2*l],lo)+dot(values[2*l+1],hi);
            sub+=scale*part;
        }
        sum[output]+=d*sub;
        }
    }
    return sum*0.25f;
}

inline float4 dot_q2_rows(device const uchar*row,device const float*x,uint cols,uint stride,uint valid,uint lane) {
 float4 sum=0.0f;
 for(uint base=0;base<cols;base+=64) {
  float y0=x[base+lane],y1=x[base+lane+32];
  #pragma unroll
  for(uint r=0;r<4;++r) {
   if(r>=valid)continue;
   device const uchar*b=row+ulong(r)*stride+(base/64)*18;
   float d=half_at(b);uint q0=(b[2+lane/4]>>(2*(lane%4)))&3,q1=(b[10+lane/4]>>(2*(lane%4)))&3;
   sum[r]+=(d*(int(q0)-1))*y0;
   sum[r]+=(d*(int(q1)-1))*y1;
  }
 }
 return sum;
}
inline float dot_quant_row(device const uchar*row,device const float*x,uint cols,uint lane) {
    if(projection_type==16 || projection_type==22)return dot_iq2_blocks(row,x,cols,lane);
    return dot_quant_rows(row,x,cols,lane,0,1).x;
}

kernel void quant_matvec(device const uchar *w [[buffer(0)]],
                         device const float *x [[buffer(1)]],
                         device float *y [[buffer(2)]],
                         constant uint *p [[buffer(3)]],
                         uint2 group [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint simd [[simdgroup_index_in_threadgroup]]) {
    uint output_row = group.x*4+simd; if (output_row >= p[1]) return;
    uint cols=p[0], rows=p[1], stride=p[2], input_stride=p[4];
    device const uchar *row = w + (ulong(group.y)*rows + output_row)*stride;
    float sum = dot_quant_row(row, x+group.y*input_stride, cols, lane);
    sum = simd_sum(sum);
    if (lane == 0) y[group.y*rows+output_row] = sum;
}

kernel void swiglu(device const float *gate [[buffer(0)]],
                   device const float *up [[buffer(1)]],
                   device float *out [[buffer(2)]],
                   constant uint *p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i < p[0]) out[i] = (gate[i] / (1.0f + exp(-gate[i]))) * up[i];
}

kernel void combine(device const float *routed [[buffer(0)]],
                    device const float *shared [[buffer(1)]],
                    device const float *weights [[buffer(2)]],
                    device const float *shared_logit [[buffer(3)]],
                    device float *out [[buffer(4)]],
                    constant uint *p [[buffer(5)]], uint i [[thread_position_in_grid]]) {
    if (i >= p[0]) return;
    float sum = 0.0f;
    for (uint e=0; e<p[1]; ++e) sum += routed[e*p[0]+i] * weights[e];
    out[i] = sum + shared[i] / (1.0f + exp(-shared_logit[0]));
}

inline float4 dot_iq4_shared(device const uchar*row,device const float*x,uint cols,uint lane,uint stride,uint count,threadgroup const float*lut) {
 float4 sum=0.0f;
 for(uint base=0;base<cols;base+=256) {
  device const uchar*b=row+(base/256)*136;float d=half_at(b);uint high=uint(b[2])|(uint(b[3])<<8);
  #pragma unroll
  for(uint g=0;g<8;++g) {
   int scale=int(((b[4+g/2]>>(4*(g%2)))&15)|(((high>>(2*g))&3)<<4))-32;
   uint q=(b[8+g*16+lane%16]>>(4*(lane/16)))&15;
   sum+=(d*scale*lut[q])*quant_inputs4(x,base+lane+32*g,stride,count);
  }
 }
 return sum;
}
