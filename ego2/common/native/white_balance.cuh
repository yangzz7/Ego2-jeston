#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
namespace ego2_white_balance {
// Exact 1280x1024 -> 160x120 INTER_AREA sample: x scale is exactly 8.
// Explicit separate multiply/add matches OpenCV's float32 area accumulation.
__global__ void areaSample(const unsigned char* src,unsigned char* dst,const int* rows,const float* weights) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=160*120*3)return;
    int c=i%3,x=(i/3)%160,y=i/(160*3);float total=0;
    for(int j=0;j<10;++j) {
        int sy=rows[y*10+j];if(sy<0)break;int sum=0;
        for(int dx=0;dx<8;++dx)sum+=src[(sy*1280+x*8+dx)*3+c];
        total=__fadd_rn(total,__fmul_rn(float(sum)*.125f,weights[y*10+j]));
    }
    dst[i]=max(0,min(255,__float2int_rn(total)));
}
// Sum uint8 area samples exactly, then use the original float64 gray-world
// gain and truncating uint8 LUT. No CPU round trip or approximate image gains.
__global__ void sampleLut(const unsigned char* sample,unsigned char* lut) {
    __shared__ unsigned int sums[3][256];
    __shared__ double gains[3];
    int t=threadIdx.x;
    for(int c=0;c<3;++c) {
        unsigned int sum=0;
        for(int i=t;i<160*120;i+=256)sum+=sample[i*3+c];
        sums[c][t]=sum;
    }
    __syncthreads();
    for(int step=128;step;step/=2) {
        if(t<step)for(int c=0;c<3;++c)sums[c][t]+=sums[c][t+step];
        __syncthreads();
    }
    if(t==0) {
        double means[3];
        for(int c=0;c<3;++c)means[c]=__ddiv_rn(double(sums[c][0]),19200.0);
        double average=__ddiv_rn(__dadd_rn(__dadd_rn(means[0],means[1]),means[2]),3.0);
        for(int c=0;c<3;++c)gains[c]=fmin(2.0,fmax(.5,__ddiv_rn(average,fmax(means[c],1.0))));
    }
    __syncthreads();
    for(int c=0;c<3;++c)lut[c*256+t]=static_cast<unsigned char>(fmin(255.0,__dmul_rn(double(t),gains[c])));
}
__global__ void lookup(const unsigned char* src,unsigned char* dst,const unsigned char* lut,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)dst[i]=lut[(i%3)*256+src[i]];
}
}
