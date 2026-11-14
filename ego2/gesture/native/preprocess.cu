// GPU execution of the existing uint8 LUT and OpenCV INTER_LINEAR map.
// The caller computes the original LUT and cv::convertMaps on the CPU once.
// No floating interpolation: identical 5-bit fractions, zero border and rounding.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <mutex>
#include <algorithm>
#include <cmath>
namespace {
void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void report(char* p,size_t n,const char* s) { if(p&&n) std::snprintf(p,n,"%s",s); }
struct Context {
    int w,h,ow,oh; unsigned char *input=nullptr,*output=nullptr,*lut=nullptr;
    unsigned char *hi=nullptr,*ho=nullptr,*hl=nullptr;
    unsigned char *hs=nullptr,*sample=nullptr;
    int *rows=nullptr;float* weights=nullptr;
    short2 *map=nullptr; unsigned short* fraction=nullptr;
    cudaStream_t stream=nullptr; cudaEvent_t done=nullptr; std::mutex mutex;
    ~Context() {
        cudaSetDevice(0); if(stream)cudaStreamSynchronize(stream);
        if(hi)cudaFreeHost(hi);if(ho)cudaFreeHost(ho);if(hl)cudaFreeHost(hl);
        if(hs)cudaFreeHost(hs);if(rows)cudaFree(rows);if(weights)cudaFree(weights);
        if(map)cudaFree(map);if(fraction)cudaFree(fraction);
        if(done)cudaEventDestroy(done);if(stream)cudaStreamDestroy(stream);
    }
};
void mapped(unsigned char*& host,unsigned char*& device,size_t n) {
    check(cudaHostAlloc(&host,n,cudaHostAllocMapped));check(cudaHostGetDevicePointer(&device,host,0));
}
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
__global__ void lookup(const unsigned char* src,unsigned char* dst,const unsigned char* lut,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)dst[i]=lut[(i%3)*256+src[i]];
}
__device__ int pixel(const unsigned char* src,int x,int y,int c,int w,int h) {
    return x>=0&&x<w&&y>=0&&y<h?src[(y*w+x)*3+c]:0;
}
__global__ void remap(const unsigned char* src,unsigned char* dst,const short2* map,const unsigned short* fraction,int w,int h,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    short2 xy=map[i];int f=fraction[i],fx=f&31,fy=(f>>5)&31;
    int a=(32-fx)*(32-fy),b=fx*(32-fy),c=(32-fx)*fy,d=fx*fy;
    for(int k=0;k<3;++k) {
        int v=pixel(src,xy.x,xy.y,k,w,h)*a+pixel(src,xy.x+1,xy.y,k,w,h)*b+
              pixel(src,xy.x,xy.y+1,k,w,h)*c+pixel(src,xy.x+1,xy.y+1,k,w,h)*d;
        dst[i*3+2-k]=(v+512)>>10;
    }
}
}
extern "C" void* ego2_pre_create(int w,int h,int ow,int oh,const void* xy,const void* fractions,char* error,size_t len) {
    Context* c=nullptr;
    try {
        if(w<1||h<1||ow<1||oh<1||w>8192||h>8192||ow>8192||oh>8192||bool(xy)!=bool(fractions))throw std::runtime_error("Invalid preprocessing dimensions/maps");
        check(cudaSetDevice(0));c=new Context();c->w=w;c->h=h;c->ow=ow;c->oh=oh;
        check(cudaStreamCreateWithFlags(&c->stream,cudaStreamNonBlocking));
        check(cudaEventCreateWithFlags(&c->done,cudaEventBlockingSync|cudaEventDisableTiming));
        mapped(c->hi,c->input,size_t(w)*h*3);mapped(c->ho,c->output,size_t(ow)*oh*3);mapped(c->hl,c->lut,768);
        if(w==1280&&h==1024&&!xy) {
            mapped(c->hs,c->sample,160*120*3);
            int rows[120*10];float weights[120*10];std::fill(rows,rows+120*10,-1);std::fill(weights,weights+120*10,0.f);
            const double scale=1024.0/120;
            for(int y=0;y<120;++y) {
                double begin=y*scale,end=begin+scale,width=std::min(scale,1024-begin);int j=0;
                for(int sy=int(std::floor(begin));sy<std::min(1024,int(std::ceil(end)));++sy) {
                    double overlap=std::min(end,double(sy+1))-std::max(begin,double(sy));
                    if(overlap>1e-3){rows[y*10+j]=sy;weights[y*10+j]=float(overlap/width);++j;}
                }
            }
            check(cudaMalloc(&c->rows,sizeof(rows)));check(cudaMalloc(&c->weights,sizeof(weights)));
            check(cudaMemcpy(c->rows,rows,sizeof(rows),cudaMemcpyHostToDevice));check(cudaMemcpy(c->weights,weights,sizeof(weights),cudaMemcpyHostToDevice));
        }
        if(xy) {
            check(cudaMalloc(&c->map,size_t(ow)*oh*sizeof(short2)));
            check(cudaMalloc(&c->fraction,size_t(ow)*oh*sizeof(unsigned short)));
            check(cudaMemcpy(c->map,xy,size_t(ow)*oh*sizeof(short2),cudaMemcpyHostToDevice));
            check(cudaMemcpy(c->fraction,fractions,size_t(ow)*oh*sizeof(unsigned short),cudaMemcpyHostToDevice));
        }
        report(error,len,"");return c;
    } catch(const std::exception& e){report(error,len,e.what());delete c;return nullptr;}
}
extern "C" void ego2_pre_destroy(void* p){delete static_cast<Context*>(p);}
extern "C" int ego2_pre_run(void* ptr,const void* input,void* output,const void* lut,char* error,size_t len) {
    auto* c=static_cast<Context*>(ptr);if(!c){report(error,len,"Preprocessor closed");return -1;}
    std::lock_guard<std::mutex> guard(c->mutex);
    try {
        if(!input||!output||(!c->map&&!lut)||(lut&&(c->map||c->ow!=c->w||c->oh!=c->h)))throw std::runtime_error("Invalid preprocessing buffers/operation");
        check(cudaSetDevice(0));std::memcpy(c->hi,input,size_t(c->w)*c->h*3);
        int count=c->ow*c->oh;
        if(lut){std::memcpy(c->hl,lut,768);lookup<<<(count*3+255)/256,256,0,c->stream>>>(c->input,c->output,c->lut,count*3);}
        else remap<<<(count+255)/256,256,0,c->stream>>>(c->input,c->output,c->map,c->fraction,c->w,c->h,count);
        check(cudaGetLastError());check(cudaEventRecord(c->done,c->stream));check(cudaEventSynchronize(c->done));
        std::memcpy(output,c->ho,size_t(count)*3);report(error,len,"");return 0;
    } catch(const std::exception& e){cudaStreamSynchronize(c->stream);report(error,len,e.what());return -1;}
}

extern "C" int ego2_pre_sample(void* ptr,const void* input,void* sample,char* error,size_t len) {
    auto* c=static_cast<Context*>(ptr);if(!c){report(error,len,"Preprocessor closed");return -1;}
    std::lock_guard<std::mutex> guard(c->mutex);
    try {
        if(!input||!sample||!c->sample)throw std::runtime_error("Area sample requires 1280x1024 LUT context");
        check(cudaSetDevice(0));std::memcpy(c->hi,input,size_t(c->w)*c->h*3);
        areaSample<<<(160*120*3+255)/256,256,0,c->stream>>>(c->input,c->sample,c->rows,c->weights);
        check(cudaGetLastError());check(cudaEventRecord(c->done,c->stream));check(cudaEventSynchronize(c->done));
        std::memcpy(sample,c->hs,160*120*3);report(error,len,"");return 0;
    }catch(const std::exception& e){cudaStreamSynchronize(c->stream);report(error,len,e.what());return -1;}
}
