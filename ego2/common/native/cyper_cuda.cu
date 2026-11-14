// Integer bilinear interpolation, with the same rounding and replicated output
// border as cv::COLOR_Bayer{RG,BG,GR,GB}2BGR. No display ISP/tone processing.
#include "cyper_cuda.h"
#include "white_balance.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <mutex>
#include <cstring>

namespace {
void check(cudaError_t e) {
    if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void report(char* p, size_t n, const char* s) { if(p && n) std::snprintf(p,n,"%s",s); }
struct Decoder {
    int w, h;
    unsigned char *raw=nullptr, *bgr=nullptr, *host_raw=nullptr, *host_bgr=nullptr;
    unsigned char *wb_sample=nullptr, *wb_lut=nullptr;
    int* wb_rows=nullptr;float* wb_weights=nullptr;
    cudaStream_t stream=nullptr;
    cudaEvent_t complete=nullptr;
    std::mutex mutex;
    Decoder(int width, int height):w(width),h(height) {}
    ~Decoder() {
        cudaSetDevice(0);
        if(stream) cudaStreamSynchronize(stream);
        if(wb_sample)cudaFree(wb_sample);if(wb_lut)cudaFree(wb_lut);
        if(wb_rows)cudaFree(wb_rows);if(wb_weights)cudaFree(wb_weights);
        if(host_raw) cudaFreeHost(host_raw);
        if(host_bgr) cudaFreeHost(host_bgr);
        if(complete) cudaEventDestroy(complete);
        if(stream) cudaStreamDestroy(stream);
    }
};
__device__ int sample(const unsigned char* p,int w,int h,int x,int y,int lane) {
    // Virtual replacement preserves the original transport and metadata bytes.
    if(y==h-1) y=h-3;
    return p[(static_cast<size_t>(y)*w+x)*2+lane];
}
__device__ int color(int x,int y,int phase) {
    if(phase>=2) x^=1;
    if((x^y)&1) return 1;
    int c=(y&1)?2:0;
    return (phase&1)?2-c:c;
}
__global__ void convert(const unsigned char* raw,unsigned char* bgr,int w,int h,int p0,int p1) {
    int ox=blockIdx.x*blockDim.x+threadIdx.x, oy=blockIdx.y*blockDim.y+threadIdx.y;
    int lane=blockIdx.z;
    if(ox>=w || oy>=h) return;
    auto* out=bgr+(static_cast<size_t>(lane)*w*h+static_cast<size_t>(oy)*w+ox)*3;
    if(w<3 || h<3) { out[0]=out[1]=out[2]=0; return; }
    // OpenCV copies the nearest fully interpolated pixel into its outer border.
    int x=max(1,min(ox,w-2)), y=max(1,min(oy,h-2));
    int phase=lane?p1:p0;
    int c=color(x,y,phase);
    int mid=sample(raw,w,h,x,y,lane);
    int lr=sample(raw,w,h,x-1,y,lane)+sample(raw,w,h,x+1,y,lane);
    int ud=sample(raw,w,h,x,y-1,lane)+sample(raw,w,h,x,y+1,lane);
    if(c==1) {
        out[1]=mid;
        int horizontal=color(x-1,y,phase);
        out[horizontal]=(lr+1)>>1;
        out[2-horizontal]=(ud+1)>>1;
    } else {
        out[c]=mid;
        out[1]=(lr+ud+2)>>2;
        int corners=sample(raw,w,h,x-1,y-1,lane)+sample(raw,w,h,x+1,y-1,lane)
                   +sample(raw,w,h,x-1,y+1,lane)+sample(raw,w,h,x+1,y+1,lane);
        out[2-c]=(corners+2)>>2;
    }
}
}
extern "C" void* ego2_cyper_create(int w,int h,char* error,size_t n) {
    Decoder* d=nullptr;
    try {
        if(w<2 || h<3 || w>8192 || h>8192) throw std::runtime_error("Invalid RAW dimensions (2..8192 x 3..8192)");
        check(cudaSetDevice(0));
        d=new Decoder(w,h);
        check(cudaStreamCreateWithFlags(&d->stream,cudaStreamNonBlocking));
        check(cudaEventCreateWithFlags(&d->complete,cudaEventBlockingSync|cudaEventDisableTiming));
        // Jetson CPU/GPU share physical DRAM. Reuse mapped staging buffers
        // instead of serializing each eye through a separate DMA transfer.
        check(cudaHostAlloc(&d->host_raw,static_cast<size_t>(w)*h*2,cudaHostAllocMapped));
        check(cudaHostAlloc(&d->host_bgr,static_cast<size_t>(w)*h*6,cudaHostAllocMapped));
        check(cudaHostGetDevicePointer(&d->raw,d->host_raw,0));
        check(cudaHostGetDevicePointer(&d->bgr,d->host_bgr,0));
        report(error,n,""); return d;
    } catch(const std::exception& e) { report(error,n,e.what()); delete d; return nullptr; }
}
extern "C" void ego2_cyper_destroy(void* ptr) { delete static_cast<Decoder*>(ptr); }
static int decode_impl(void* ptr,const unsigned char* raw,size_t input_step,
    int p0,int p1,unsigned char* a,size_t sa,unsigned char* b,size_t sb,char* error,size_t n,bool balance) {
    auto* d=static_cast<Decoder*>(ptr);
    if(!d) { report(error,n,"Decoder is closed"); return -1; }
    std::lock_guard<std::mutex> lock(d->mutex);
    try {
        if(!raw || !a || !b || input_step<size_t(d->w)*2 || sa<size_t(d->w)*3 || sb<size_t(d->w)*3 ||
           p0<0 || p0>3 || p1<0 || p1>3) throw std::runtime_error("Invalid RAW pointers, row stride or Bayer phase");
        check(cudaSetDevice(0));
        size_t row=static_cast<size_t>(d->w)*3, plane=row*d->h;
        if(input_step==size_t(d->w)*2) std::memcpy(d->host_raw,raw,size_t(d->w)*d->h*2);
        else for(int y=0;y<d->h;++y) std::memcpy(d->host_raw+size_t(y)*d->w*2,raw+size_t(y)*input_step,d->w*2);
        convert<<<dim3((d->w+31)/32,(d->h+7)/8,2),dim3(32,8),0,d->stream>>>(d->raw,d->bgr,d->w,d->h,p0,p1);
        check(cudaGetLastError());
        if(balance) {
            if(d->w!=1280||d->h!=1024)throw std::runtime_error("Fused white balance requires 1280x1024");
            if(!d->wb_sample) {
                // Allocate all resources before publishing the initialized marker.
                // A failed initialization makes this decoder fail closed.
                check(cudaMalloc(&d->wb_rows,120*10*sizeof(int)));
                check(cudaMalloc(&d->wb_weights,120*10*sizeof(float)));
                check(cudaMalloc(&d->wb_lut,768));
                int rows[120*10];float weights[120*10];
                std::fill(rows,rows+120*10,-1);std::fill(weights,weights+120*10,0.f);
                const double scale=1024.0/120;
                for(int y=0;y<120;++y) {
                    double begin=y*scale,end=begin+scale,width=std::min(scale,1024-begin);int j=0;
                    for(int sy=int(std::floor(begin));sy<std::min(1024,int(std::ceil(end)));++sy) {
                        double overlap=std::min(end,double(sy+1))-std::max(begin,double(sy));
                        if(overlap>1e-3){rows[y*10+j]=sy;weights[y*10+j]=float(overlap/width);++j;}
                    }
                }
                check(cudaMemcpy(d->wb_rows,rows,sizeof(rows),cudaMemcpyHostToDevice));
                check(cudaMemcpy(d->wb_weights,weights,sizeof(weights),cudaMemcpyHostToDevice));
                check(cudaMalloc(&d->wb_sample,160*120*3));
            }
            for(int eye=0;eye<2;++eye) {
                unsigned char* image=d->bgr+size_t(eye)*plane;
                ego2_white_balance::areaSample<<<(160*120*3+255)/256,256,0,d->stream>>>(image,d->wb_sample,d->wb_rows,d->wb_weights);
                check(cudaGetLastError());
                ego2_white_balance::sampleLut<<<1,256,0,d->stream>>>(d->wb_sample,d->wb_lut);
                check(cudaGetLastError());
                ego2_white_balance::lookup<<<(plane+255)/256,256,0,d->stream>>>(image,image,d->wb_lut,int(plane));
                check(cudaGetLastError());
            }
        }
        // A blocking event lets the capture CPU sleep while the GPU computes.
        check(cudaEventRecord(d->complete,d->stream));
        check(cudaEventSynchronize(d->complete));
        if(sa==row) std::memcpy(a,d->host_bgr,plane);
        else for(int y=0;y<d->h;++y) std::memcpy(a+size_t(y)*sa,d->host_bgr+size_t(y)*row,row);
        if(sb==row) std::memcpy(b,d->host_bgr+plane,plane);
        else for(int y=0;y<d->h;++y) std::memcpy(b+size_t(y)*sb,d->host_bgr+plane+size_t(y)*row,row);
        report(error,n,""); return 0;
    } catch(const std::exception& e) {
        cudaStreamSynchronize(d->stream); // No outstanding writes after error return.
        report(error,n,e.what()); return -1;
    }
}

extern "C" int ego2_cyper_decode(void* ptr,const unsigned char* raw,size_t step,int p0,int p1,
    unsigned char* a,size_t sa,unsigned char* b,size_t sb,char* error,size_t n) {
    return decode_impl(ptr,raw,step,p0,p1,a,sa,b,sb,error,n,false);
}
extern "C" int ego2_cyper_decode_balanced(void* ptr,const unsigned char* raw,size_t step,int p0,int p1,
    unsigned char* a,size_t sa,unsigned char* b,size_t sb,char* error,size_t n) {
    return decode_impl(ptr,raw,step,p0,p1,a,sa,b,sb,error,n,true);
}
