#pragma once
#include "cyper_cuda.h"
#include <dlfcn.h>
#include <unistd.h>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <opencv2/core.hpp>

namespace ego2 {
inline bool cuda_capture_enabled() {
    const char* setting=std::getenv("EGO2_CAPTURE_BACKEND");
    std::string backend=setting?setting:(access("/etc/nv_tegra_release",F_OK)==0?"cuda":"cpu");
    if(backend!="cpu" && backend!="cuda") throw std::runtime_error("EGO2_CAPTURE_BACKEND must be cuda or cpu");
    return backend=="cuda";
}
class CudaCapture {
    int width_, height_;
    void *library_=nullptr, *decoder_=nullptr;
    decltype(&ego2_cyper_destroy) destroy_=nullptr;
    decltype(&ego2_cyper_decode) decode_=nullptr;
public:
    CudaCapture(int width,int height):width_(width),height_(height) {
        const char* override_path=std::getenv("EGO2_CAPTURE_CUDA_LIBRARY");
        std::string path;
        if(override_path) path=override_path;
        else {
            char exe[4096]; auto n=readlink("/proc/self/exe",exe,sizeof(exe)-1);
            if(n<0) throw std::runtime_error("Cannot locate CUDA capture library");
            exe[n]=0; path=exe;
            // root/build/reconstruction/bin/executable -> root/build
            for(int i=0;i<3;++i) path=path.substr(0,path.find_last_of('/'));
            path+="/cyper_capture/libego2_cyper_cuda.so";
        }
        try {
            library_=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
            if(!library_) throw std::runtime_error(std::string("CUDA capture library: ")+dlerror());
            auto create=reinterpret_cast<decltype(&ego2_cyper_create)>(dlsym(library_,"ego2_cyper_create"));
            destroy_=reinterpret_cast<decltype(destroy_)>(dlsym(library_,"ego2_cyper_destroy"));
            decode_=reinterpret_cast<decltype(decode_)>(dlsym(library_,"ego2_cyper_decode"));
            if(!create || !destroy_ || !decode_) throw std::runtime_error("Incompatible CUDA capture library ABI");
            char error[1024]; decoder_=create(width,height,error,sizeof(error));
            if(!decoder_) throw std::runtime_error(std::string("CUDA capture initialization: ")+error);
        } catch(...) { if(library_) dlclose(library_); throw; }
    }
    CudaCapture(const CudaCapture&)=delete;
    CudaCapture& operator=(const CudaCapture&)=delete;
    ~CudaCapture() { if(decoder_) destroy_(decoder_); if(library_) dlclose(library_); }
    void decode(const cv::Mat& raw, int firmware, cv::Mat& cam0, cv::Mat& cam1) {
        if(raw.type()!=CV_8UC2 || raw.cols!=width_ || raw.rows!=height_ || firmware<3 || firmware>7)
            throw std::runtime_error("Invalid S2 CUDA input");
        cam0.create(raw.size(),CV_8UC3); cam1.create(raw.size(),CV_8UC3);
        char error[1024];
        if(decode_(decoder_,raw.data,raw.step,firmware>=4?1:0,0,cam0.data,cam0.step,
                   cam1.data,cam1.step,error,sizeof(error)))
            throw std::runtime_error(std::string("CUDA capture conversion: ")+error);
    }
};
}
