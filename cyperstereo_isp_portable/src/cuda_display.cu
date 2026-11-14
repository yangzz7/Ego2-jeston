// CUDA display implementation of the existing Apache-2.0 tunable ISP stages.
// Integer gates, CFA phases, gain adaptation and tone parameters match the CPU
// pipeline. AWB estimation/history remains shared host code. No camera access.
#include "cuda_display.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace camera_tuner {
namespace {
void checked(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(std::string("CUDA display: ")+cudaGetErrorString(status));
}
__device__ inline void ClipMinMax(int& v, int hi, int lo) { v=max(lo,min(hi,v)); }
__device__ inline int AbsInt(int value)
{
    return value < 0 ? -value : value;
}

__device__ inline int RoundShift8Signed(int value)
{
    return value >= 0 ? (value + 128) >> 8 : -((-value + 128) >> 8);
}

__device__ inline int RoundShift4Signed(int value)
{
    return value >= 0 ? (value + 8) >> 4 : -((-value + 8) >> 4);
}

__device__ inline int RoundDivSigned(int value, int divisor)
{
    return value >= 0 ? (value + divisor / 2) / divisor
                      : -((-value + divisor / 2) / divisor);
}

__device__ inline int Luma601(const int32_t *pixel)
{
    const int b = pixel[0];
    const int g = pixel[1];
    const int r = pixel[2];
    return (29 * b + 150 * g + 77 * r + 128) >> 8;
}

// Fast hue guard. If a filtered saturated colour would rotate farther than
// the configured angle, retain the original chroma vector. This is stricter
// and considerably cheaper than projecting every pixel with 64-bit divides.
__device__ inline bool HueWithinLimit(int dr0, int db0, int tan_limit_q8,
                           int dr, int db)
{
    const int64_t dot = static_cast<int64_t>(dr0) * dr +
                        static_cast<int64_t>(db0) * db;
    if (dot <= 0)
        return false;
    const int64_t cross = static_cast<int64_t>(dr0) * db -
                          static_cast<int64_t>(db0) * dr;
    const int64_t abs_cross = cross < 0 ? -cross : cross;
    return abs_cross * 256 <= dot * tan_limit_q8;
}

// Scale all B/G/R offsets from the same luminance by one factor when the
// reconstructed colour leaves the valid gamut. Independent channel clipping
// would rotate hue at saturated colours.
__device__ inline void GamutClampPreserveHue(int y, int max_value, int &b, int &g, int &r)
{
    int scale_q8 = 256;
    const int channels[3] = {b, g, r};
    for (int i = 0; i < 3; ++i) {
        const int delta = channels[i] - y;
        if (channels[i] > max_value && delta > 0) {
            scale_q8 = min(scale_q8,
                static_cast<int>(static_cast<int64_t>(max_value - y) * 256 /
                                 delta));
        } else if (channels[i] < 0 && delta < 0) {
            scale_q8 = min(scale_q8,
                static_cast<int>(static_cast<int64_t>(y) * 256 / -delta));
        }
    }

    if (scale_q8 < 256) {
        b = y + RoundShift8Signed((b - y) * scale_q8);
        g = y + RoundShift8Signed((g - y) * scale_q8);
        r = y + RoundShift8Signed((r - y) * scale_q8);
    }
    ClipMinMax(b, max_value, 0);
    ClipMinMax(g, max_value, 0);
    ClipMinMax(r, max_value, 0);
}


__device__ int phase(int x, int y, bool rggb) {
    // 0=R, 1=GB (red horizontal), 2=GR (blue horizontal), 3=B.
    int p=((y&1)<<1)|(x&1); return rggb ? p : 3-p;
}
__device__ int avg2(int a,int b) { return RoundDivSigned(a+b,2); }
__device__ int avg4(int a,int b,int c,int d) { return RoundDivSigned(a+b+c+d,4); }
#define PIXEL(W,H) const int x=blockIdx.x*blockDim.x+threadIdx.x; const int y=blockIdx.y*blockDim.y+threadIdx.y; if(x>=(W)||y>=(H)) return; const int i=y*(W)+x
__global__ void blackKernel(const uint8_t* raw,int* out,int w,int h,int black) {
    PIXEL(w,h); out[i]=max(0,(int(raw[i])<<2)-black);
}
__global__ void bayerKernel(const int* src,int* dst,int w,int h,CudaDisplayParameters const* cfg) {
    PIXEL(w,h); const auto p=cfg->bayer;
    int center=src[i], value=center;
    if (x>=2 && x<w-2 && y>=2 && y<h-2) {
        int threshold=max(1,p.threshold_base+((center*p.threshold_signal_q8+128)>>8));
        int sum=p.center_weight*center, weight=p.center_weight;
        const int offsets[4]={-2,2,-2*w,2*w};
        for(int o:offsets) { int s=src[i+o]; if(abs(s-center)<=threshold){sum+=s; ++weight;} }
        if(weight!=p.center_weight) {
            const int reciprocal[9]={0,65536,32768,21845,16384,13107,10923,9362,8192};
            int filtered=(static_cast<int64_t>(sum)*reciprocal[weight]+32768)>>16;
            value=max(0,min(1023,center+RoundShift8Signed((filtered-center)*p.strength_q8)));
        }
    }
    int ph=phase(x,y,cfg->rggb);
    float gain=cfg->wb[ph==0 ? 0 : ph==3 ? 2 : 1];
    dst[i]=max(0,min(1023,int(value*gain)));
}
__global__ void greenKernel(const int* raw,int* green,int w,int h,int rggb) {
    PIXEL(w,h); int ph=phase(x,y,rggb);
    if(ph==1 || ph==2) { green[i]=raw[i]; return; }
    int g;
    if(x>=2&&x<w-2&&y>=2&&y<h-2) {
        int center=raw[i], gh=(raw[i-1]+raw[i+1]+1)>>1, gv=(raw[i-w]+raw[i+w]+1)>>1;
        int dh=abs(raw[i-1]-raw[i+1])+abs(2*center-raw[i-2]-raw[i+2]);
        int dv=abs(raw[i-w]-raw[i+w])+abs(2*center-raw[i-2*w]-raw[i+2*w]);
        g=dh<dv?gh:dv<dh?gv:(gh+gv+1)>>1;
    } else {
        int s=0,n=0;
        if(x>0){s+=raw[i-1];++n;} if(x+1<w){s+=raw[i+1];++n;}
        if(y>0){s+=raw[i-w];++n;} if(y+1<h){s+=raw[i+w];++n;}
        g=(s+n/2)/n;
    }
    green[i]=max(0,min(1023,g));
}
__global__ void demosaicKernel(const int* raw,const int* green,int* bgr,int16_t* luma,int16_t* dr,int16_t* db,int w,int h,CudaDisplayParameters const* cfg) {
    PIXEL(w,h);
    int xx=max(1,min(w-2,x)), yy=max(1,min(h-2,y)), j=yy*w+xx;
    int g=green[j],r=g,b=g, ph=phase(xx,yy,cfg->rggb);
    if(ph==0 || ph==3) {
        int v=g+avg4(raw[j-w-1]-green[j-w-1],raw[j-w+1]-green[j-w+1],
                     raw[j+w-1]-green[j+w-1],raw[j+w+1]-green[j+w+1]);
        if(ph==0){r=raw[j];b=v;}else{b=raw[j];r=v;}
    } else {
        int horizontal=g+avg2(raw[j-1]-green[j-1],raw[j+1]-green[j+1]);
        int vertical=g+avg2(raw[j-w]-green[j-w],raw[j+w]-green[j+w]);
        if(ph==2){b=horizontal;r=vertical;}else{r=horizontal;b=vertical;}
    }
    b=max(0,min(1023,b)); r=max(0,min(1023,r));
    for(int c=0;c<3;++c) bgr[3*i+2-c]=max(0,min(1023,int(r*cfg->ccm[3*c]+g*cfg->ccm[3*c+1]+b*cfg->ccm[3*c+2])));
    luma[i]=Luma601(bgr+3*i);dr[i]=bgr[3*i+2]-bgr[3*i+1];db[i]=bgr[3*i]-bgr[3*i+1];
}
__global__ void lumaOnlyKernel(const int* src,int16_t* luma,int w,int h) {
    PIXEL(w,h); luma[i]=Luma601(src+3*i);
}
__global__ void quarterKernel(const int16_t* luma,int* q,int w,int h) {
    PIXEL(w/4,h/4); int s=0;
    for(int dy=0;dy<4;++dy)for(int dx=0;dx<4;++dx)s+=luma[(4*y+dy)*w+4*x+dx];
    q[i]=__float2int_rn(float(s)*(1.f/16));
}
__device__ int reflect(int x,int n) {return x<0?-x-1:x>=n?2*n-x-1:x;}
__global__ void guideKernel(const int* q,int* qa,int* qb,int qw,int qh,int eps) {
    PIXEL(qw,qh); float s=0,ss=0;
    for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){float v=q[reflect(y+dy,qh)*qw+reflect(x+dx,qw)];s+=v;ss+=v*v;}
    float mean=s*(1.f/9), variance=fmaxf(0.f,ss*(1.f/9)-mean*mean);
    float coeff=variance/(variance+eps);
    qa[i]=int(coeff*4096.f+0.5f);qb[i]=int((1.f-coeff)*mean*16.f+0.5f);
}
__global__ void guideMeanKernel(const int* a,const int* b,int* ma,int* mb,int qw,int qh) {
    PIXEL(qw,qh); int sa=0,sb=0;
    for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx){int j=reflect(y+dy,qh)*qw+reflect(x+dx,qw);sa+=a[j];sb+=b[j];}
    ma[i]=__float2int_rn(sa*(1.f/9)); mb[i]=__float2int_rn(sb*(1.f/9));
}
__global__ void lumaKernel(const int* src,int* dst,const int16_t* luma,const int* ma,const int* mb,int w,int h,int strength,int16_t* outLuma,int16_t* dr,int16_t* db) {
    PIXEL(w,h); int q=(y/4)*(w/4)+x/4,y0=luma[i];
    int yf=max(0,min(1023,(ma[q]*y0+(mb[q]<<8)+2048)>>12));
    int b=src[3*i],g=src[3*i+1],r=src[3*i+2];
    int delta=RoundShift8Signed((yf-y0)*strength);
    delta=max(-min(b,min(g,r)),min(delta,1023-max(b,max(g,r))));
    dst[3*i]=b+delta;dst[3*i+1]=g+delta;dst[3*i+2]=r+delta;
    outLuma[i]=Luma601(dst+3*i);dr[i]=r-g;db[i]=b-g;
}
__global__ void chromaKernel(const int32_t* src,int32_t* dst,const int16_t* luma,const int16_t* dr_plane,const int16_t* db_plane,int width,int height,ChromaNrPrms p) {
    PIXEL(width,height);
    for(int c=0;c<3;++c)dst[3*i+c]=src[3*i+c];
    if(x<1||x>=width-1||y<1||y>=height-1)return;
    int x_pos=x,row=y*width,ty=max(p.luma_threshold,1),tc=max(p.chroma_threshold,1),sat_low=max(p.saturation_low,0),max_value=1023;
            const int pixel_idx = row + x_pos;

            const int y0 = luma[pixel_idx];
            const int dr0 = dr_plane[pixel_idx];
            const int db0 = db_plane[pixel_idx];
            const int left = pixel_idx - 1;
            const int right = pixel_idx + 1;
            const int up = pixel_idx - width;
            const int down = pixel_idx + width;

            const int up_left = up - 1;
            const int up_right = up + 1;
            const int down_left = down - 1;
            const int down_right = down + 1;

            // Shared 3x3 kernel for Dr and Db. Keep the neighbour-only sum as
            // well: it is a robust local colour base for isolated speckles,
            // while the ordinary filtered candidate retains the centre weight.
            const int neighbor_dr_sum =
                2 * (static_cast<int>(dr_plane[left]) + dr_plane[right] +
                     dr_plane[up] + dr_plane[down]) +
                dr_plane[up_left] + dr_plane[up_right] +
                dr_plane[down_left] + dr_plane[down_right];
            const int neighbor_db_sum =
                2 * (static_cast<int>(db_plane[left]) + db_plane[right] +
                     db_plane[up] + db_plane[down]) +
                db_plane[up_left] + db_plane[up_right] +
                db_plane[down_left] + db_plane[down_right];
            const int filtered_dr = RoundShift4Signed(
                4 * dr0 + neighbor_dr_sum);
            const int filtered_db = RoundShift4Signed(
                4 * db0 + neighbor_db_sum);

            // Special case for the failure mode seen on white walls at high
            // gain. The original chroma-delta guard rejected large isolated
            // deviations, preserving exactly the red/blue speckles we want to
            // remove. Only allow the stronger correction when all surrounding
            // pixels are luma-flat and their common colour is near-neutral.
            const int base_dr = RoundDivSigned(neighbor_dr_sum, 12);
            const int base_db = RoundDivSigned(neighbor_db_sum, 12);
            const int base_chroma = AbsInt(base_dr) + AbsInt(base_db);
            const int outlier = AbsInt(dr0 - base_dr) + AbsInt(db0 - base_db);
            if (base_chroma <= max(p.neutral_base_chroma, 0) &&
                outlier >= max(p.neutral_outlier_threshold, 1)) {
                const int neighbours[8] = {
                    left, right, up, down,
                    up_left, up_right, down_left, down_right};
                int luma_min = luma[neighbours[0]];
                int luma_max = luma_min;
                for (int n = 1; n < 8; ++n) {
                    const int value = luma[neighbours[n]];
                    luma_min = min(luma_min, value);
                    luma_max = max(luma_max, value);
                }
                const int neighbor_luma_sum =
                    2 * (static_cast<int>(luma[left]) + luma[right] +
                         luma[up] + luma[down]) +
                    luma[up_left] + luma[up_right] +
                    luma[down_left] + luma[down_right];
                const int base_y = (neighbor_luma_sum + 6) / 12;
                const int flat_limit =
                    max(p.neutral_luma_range, 1);
                if (luma_max - luma_min <= flat_limit &&
                    AbsInt(y0 - base_y) <= flat_limit) {
                    const int alpha_q8 = max(0, min(
                        p.neutral_outlier_strength_q8, 256));
                    const int dr = dr0 + RoundShift8Signed(
                        (base_dr - dr0) * alpha_q8);
                    const int db = db0 + RoundShift8Signed(
                        (base_db - db0) * alpha_q8);
                    int g = y0 - RoundShift8Signed(77 * dr + 29 * db);
                    int r = g + dr;
                    int b = g + db;
                    GamutClampPreserveHue(y0, max_value, b, g, r);
                    dst[3 * pixel_idx + 0] = b;
                    dst[3 * pixel_idx + 1] = g;
                    dst[3 * pixel_idx + 2] = r;
                    return;
                }
            }

            const int max_luma_delta = max(
                max(AbsInt(static_cast<int>(luma[left]) - y0),
                           AbsInt(static_cast<int>(luma[right]) - y0)),
                max(AbsInt(static_cast<int>(luma[up]) - y0),
                           AbsInt(static_cast<int>(luma[down]) - y0)));
            if (max_luma_delta > ty)
                return;
            if (AbsInt(filtered_dr - dr0) + AbsInt(filtered_db - db0) > tc)
                return;

            const int saturation = AbsInt(dr0) + AbsInt(db0);
            const int alpha_q8 = saturation <= sat_low ? p.neutral_strength_q8 :
                saturation >= p.saturation_high ? p.saturated_strength_q8 :
                p.neutral_strength_q8 + (p.saturated_strength_q8-p.neutral_strength_q8)*
                (saturation-sat_low)/(p.saturation_high-sat_low);
            int dr = dr0 +
                     RoundShift8Signed((filtered_dr - dr0) * alpha_q8);
            int db = db0 +
                     RoundShift8Signed((filtered_db - db0) * alpha_q8);

            if (dr == dr0 && db == db0)
                return;
            if (saturation >= sat_low &&
                !HueWithinLimit(dr0, db0,
                                max(p.max_hue_tan_q8, 0), dr, db))
                return;

            // Reconstruct BGR while keeping the original Rec.601 luminance.
            int g = y0 - RoundShift8Signed(77 * dr + 29 * db);
            int r = g + dr;
            int b = g + db;
            GamutClampPreserveHue(y0, max_value, b, g, r);
            dst[3 * pixel_idx + 0] = b;
            dst[3 * pixel_idx + 1] = g;
            dst[3 * pixel_idx + 2] = r;
}
__global__ void falseQuarterKernel(const int32_t* src,const int16_t* luma,int16_t* q_y,int16_t* q_dr,int16_t* q_db,int16_t* q_hf,int w,int h) {
    PIXEL(w/4,h/4); int sy=0,sdr=0,sdb=0,shf=0;
    for(int dy=0;dy<4;++dy)for(int dx=0;dx<4;++dx){
        int xx=4*x+dx,yy=4*y+dy,j=yy*w+xx,v=luma[j];
        sy+=v;sdr+=src[3*j+2]-src[3*j+1];sdb+=src[3*j]-src[3*j+1];
        if(xx>0&&xx<w-1&&yy>0&&yy<h-1)shf+=(abs(4*v-luma[j-1]-luma[j+1]-luma[j-w]-luma[j+w])+2)>>2;
    }
    q_y[i]=(sy+8)/16;q_dr[i]=RoundDivSigned(sdr,16);q_db[i]=RoundDivSigned(sdb,16);q_hf[i]=(shf+8)/16;
}
__global__ void falseGuideKernel(const int16_t* q_y,const int16_t* q_dr,const int16_t* q_db,const int16_t* q_hf,int16_t* base_dr,int16_t* base_db,uint8_t* attenuation,int q_width,int q_height,FalseColorPrms p) {
    PIXEL(q_width,q_height); int qx=x,qy=y;
    base_dr[i]=base_db[i]=0;attenuation[i]=0;
    int tex_low=max(p.texture_low,0),tex_high=max(p.texture_high,tex_low+1);
    int neutral_low=max(p.neutral_chroma_low,0),neutral_high=max(p.neutral_chroma_high,neutral_low+1),max_attenuation=max(0,min(p.max_attenuation_q8,256));
            int sum_y = 0;
            int sum_hf = 0;
            int texture_count = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                const int ny = qy + dy;
                if (ny < 0 || ny >= q_height) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nx = qx + dx;
                    if (nx < 0 || nx >= q_width) continue;
                    const int ni = ny * q_width + nx;
                    sum_y += q_y[ni];
                    sum_hf += q_hf[ni];
                    ++texture_count;
                }
            }

            const int mean_y =
                (sum_y + texture_count / 2) / texture_count;
            if (mean_y < p.min_luma)
                return;
            const int texture =
                (sum_hf + texture_count / 2) / texture_count;
            if (texture <= tex_low)
                return;
            int texture_q8 = texture >= tex_high
                ? 256 : (texture - tex_low) * 256 / (tex_high - tex_low);

            // A 7x7 quarter-grid window (~28x28 full resolution) is wider
            // than the colour-beat period on grilles and roof tiles. It
            // recovers the stable surface colour even when a 12x12 mean is
            // itself orange or cyan.
            int sum_dr = 0;
            int sum_db = 0;
            int chroma_count = 0;
            for (int dy = -3; dy <= 3; ++dy) {
                const int ny = qy + dy;
                if (ny < 0 || ny >= q_height) continue;
                for (int dx = -3; dx <= 3; ++dx) {
                    const int nx = qx + dx;
                    if (nx < 0 || nx >= q_width) continue;
                    const int ni = ny * q_width + nx;
                    sum_dr += q_dr[ni];
                    sum_db += q_db[ni];
                    ++chroma_count;
                }
            }
            const int mean_dr = RoundDivSigned(sum_dr, chroma_count);
            const int mean_db = RoundDivSigned(sum_db, chroma_count);
            const int mean_chroma = AbsInt(mean_dr) + AbsInt(mean_db);
            if (mean_chroma >= neutral_high)
                return;
            const int neutral_q8 = mean_chroma <= neutral_low
                ? 256 : (neutral_high - mean_chroma) * 256 /
                            (neutral_high - neutral_low);

            int reduce_q8 =
                (max_attenuation * texture_q8 + 128) >> 8;
            reduce_q8 = (reduce_q8 * neutral_q8 + 128) >> 8;
            const int qi = qy * q_width + qx;
            base_dr[qi] = static_cast<int16_t>(mean_dr);
            base_db[qi] = static_cast<int16_t>(mean_db);
            attenuation[qi] =
                static_cast<uint8_t>(min(reduce_q8, 255));
}
__global__ void falseKernel(const int* src,int* dst,const int16_t* luma,const int16_t* base_dr,const int16_t* base_db,const uint8_t* attenuation,int w,int h) {
    PIXEL(w,h);int q=(y/4)*(w/4)+x/4,reduce=attenuation[q];
    for(int c=0;c<3;++c)dst[3*i+c]=src[3*i+c];
    if(!reduce)return;
    int dr=src[3*i+2]-src[3*i+1],db=src[3*i]-src[3*i+1];
    int nr=base_dr[q]+RoundShift8Signed((dr-base_dr[q])*(256-reduce));
    int nb=base_db[q]+RoundShift8Signed((db-base_db[q])*(256-reduce));
    int yy=luma[i],g=yy-RoundShift8Signed(77*nr+29*nb),r=g+nr,b=g+nb;
    GamutClampPreserveHue(yy,1023,b,g,r);dst[3*i]=b;dst[3*i+1]=g;dst[3*i+2]=r;
}
__global__ void encodeKernel(const int* src,uint8_t* dst,int w,int h,CudaDisplayParameters const* p) {
    PIXEL(w,h); float colors[3];
    for(int c=0;c<3;++c)colors[c]=p->encode[max(0,min(1023,src[3*i+c]))];
    float luma=.114f*colors[0]+.587f*colors[1]+.299f*colors[2];
    float t=fminf(1.f,fmaxf(0.f,(luma-p->blackPoint)/(1.f-p->blackPoint)));
    float target=(1.f-p->contrast)*t+p->contrast*t*t*(3.f-2.f*t);
    float ratio=luma>0?target/luma:0,maximum=fmaxf(colors[0],fmaxf(colors[1],colors[2]));
    if(maximum>0)ratio=fminf(ratio,1.f/maximum);
    for(int c=0;c<3;++c)dst[3*i+c]=max(0,min(255,__float2int_rn(colors[c]*ratio*255.f)));
}
} // namespace

struct CudaDisplay::Impl {
    cudaStream_t stream=nullptr;
    cudaEvent_t done=nullptr;
    std::vector<void*> allocations;
    int width=0,height=0;
    uint8_t *raw=nullptr,*output=nullptr,*hostRaw=nullptr,*hostOutput=nullptr,*atten=nullptr;
    int *a=nullptr,*b=nullptr,*green=nullptr,*rgbA=nullptr,*rgbB=nullptr,*q=nullptr,*qa=nullptr,*qb=nullptr,*ma=nullptr,*mb=nullptr;
    int16_t *luma=nullptr,*dr=nullptr,*db=nullptr,*qy=nullptr,*qdr=nullptr,*qdb=nullptr,*qh=nullptr,*baseDr=nullptr,*baseDb=nullptr;
    CudaDisplayParameters *parameters=nullptr,*hostParameters=nullptr;
    template<class T> void alloc(T*& p,size_t n) { checked(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T))); allocations.push_back(p); }
    void release() noexcept {
        if(stream)cudaStreamSynchronize(stream);
        for(void* p:allocations)cudaFree(p);allocations.clear();
        if(hostRaw)cudaFreeHost(hostRaw);if(hostOutput)cudaFreeHost(hostOutput);if(hostParameters)cudaFreeHost(hostParameters);
        hostRaw=hostOutput=nullptr;hostParameters=nullptr;width=height=0;
    }
    ~Impl(){release();if(done)cudaEventDestroy(done);if(stream)cudaStreamDestroy(stream);}
    void prepare(int w,int h) {
        if(width==w&&height==h)return;
        release();size_t n=size_t(w)*h,nq=n/16;
        try {
            alloc(a,n);alloc(b,n);alloc(green,n);alloc(rgbA,3*n);alloc(rgbB,3*n);
            alloc(luma,n);alloc(dr,n);alloc(db,n);alloc(q,nq);alloc(qa,nq);alloc(qb,nq);alloc(ma,nq);alloc(mb,nq);
            alloc(qy,nq);alloc(qdr,nq);alloc(qdb,nq);alloc(qh,nq);alloc(baseDr,nq);alloc(baseDb,nq);alloc(atten,nq);
            checked(cudaHostAlloc(reinterpret_cast<void**>(&hostRaw),n,cudaHostAllocMapped));checked(cudaHostAlloc(reinterpret_cast<void**>(&hostOutput),3*n,cudaHostAllocMapped));
            checked(cudaHostAlloc(reinterpret_cast<void**>(&hostParameters),sizeof(CudaDisplayParameters),cudaHostAllocMapped));
            checked(cudaHostGetDevicePointer(&raw,hostRaw,0));checked(cudaHostGetDevicePointer(&output,hostOutput,0));
            checked(cudaHostGetDevicePointer(&parameters,hostParameters,0));
            width=w;height=h;
        }catch(...){release();throw;}
    }
};
CudaDisplay::CudaDisplay():impl_(std::make_unique<Impl>()) {
    int count=0;checked(cudaGetDeviceCount(&count));if(!count)throw std::runtime_error("No CUDA device for display ISP");
    checked(cudaSetDevice(0));checked(cudaStreamCreateWithFlags(&impl_->stream,cudaStreamNonBlocking));
    checked(cudaEventCreateWithFlags(&impl_->done,cudaEventBlockingSync|cudaEventDisableTiming));
}
CudaDisplay::~CudaDisplay()=default;
void CudaDisplay::process(const uint8_t* input,int w,int h,const CudaDisplayParameters& params,uint8_t* result) {
    if(w<8||h<8||(w%4)||(h%4))throw std::invalid_argument("CUDA display requires dimensions divisible by four");
    checked(cudaSetDevice(0));auto& p=*impl_;p.prepare(w,h);size_t n=size_t(w)*h;
    std::memcpy(p.hostRaw,input,n);*p.hostParameters=params;
    auto s=p.stream;dim3 block(32,8),grid((w+31)/32,(h+7)/8),qgrid((w/4+31)/32,(h/4+7)/8);
    // Jetson shares DRAM: mapped staging avoids redundant DMA copies.
    // Intermediates stay device-resident; completion precedes any CPU access.
    blackKernel<<<grid,block,0,s>>>(p.raw,p.a,w,h,params.black);
    bayerKernel<<<grid,block,0,s>>>(p.a,p.b,w,h,p.parameters);
    greenKernel<<<grid,block,0,s>>>(p.b,p.green,w,h,params.rggb);
    demosaicKernel<<<grid,block,0,s>>>(p.b,p.green,p.rgbA,p.luma,p.dr,p.db,w,h,p.parameters);
    quarterKernel<<<qgrid,block,0,s>>>(p.luma,p.q,w,h);
    guideKernel<<<qgrid,block,0,s>>>(p.q,p.qa,p.qb,w/4,h/4,params.luma.eps);
    guideMeanKernel<<<qgrid,block,0,s>>>(p.qa,p.qb,p.ma,p.mb,w/4,h/4);
    lumaKernel<<<grid,block,0,s>>>(p.rgbA,p.rgbB,p.luma,p.ma,p.mb,w,h,params.luma.strength_q8,p.luma,p.dr,p.db);
    chromaKernel<<<grid,block,0,s>>>(p.rgbB,p.rgbA,p.luma,p.dr,p.db,w,h,params.chroma);
    lumaOnlyKernel<<<grid,block,0,s>>>(p.rgbA,p.luma,w,h);
    falseQuarterKernel<<<qgrid,block,0,s>>>(p.rgbA,p.luma,p.qy,p.qdr,p.qdb,p.qh,w,h);
    falseGuideKernel<<<qgrid,block,0,s>>>(p.qy,p.qdr,p.qdb,p.qh,p.baseDr,p.baseDb,p.atten,w/4,h/4,params.falseColor);
    falseKernel<<<grid,block,0,s>>>(p.rgbA,p.rgbB,p.luma,p.baseDr,p.baseDb,p.atten,w,h);
    encodeKernel<<<grid,block,0,s>>>(p.rgbB,p.output,w,h,p.parameters);
    checked(cudaGetLastError());
    checked(cudaEventRecord(p.done,s));checked(cudaEventSynchronize(p.done));
    std::memcpy(result,p.hostOutput,3*n);
}
} // namespace camera_tuner
