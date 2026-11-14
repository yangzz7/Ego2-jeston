#pragma once
#include <sys/file.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <opencv2/core.hpp>

// Lossless cam0,cam1 BGR transport. flock protects readers from torn copies;
// the sequence in each slot rejects notifications overwritten by a slow reader.
class FrameRing {
    int fd_ = -1;
    unsigned char* data_ = nullptr;
    bool with_raw_;
    size_t stride_;
public:
    static constexpr size_t image_bytes = 1280u*1024u*3u;
    static constexpr size_t slot_bytes = sizeof(uint64_t)+2*image_bytes;
    static constexpr size_t raw_bytes = 1280u*1024u*2u;
    static constexpr size_t preview_slot_bytes = slot_bytes+sizeof(uint64_t)+raw_bytes;
    static constexpr size_t slots = 3;
    explicit FrameRing(const std::string& path, bool with_raw=false)
        : with_raw_(with_raw), stride_(with_raw?preview_slot_bytes:slot_bytes) {
        fd_=open(path.c_str(),O_RDWR|O_CREAT|O_EXCL|O_CLOEXEC,0600);
        if(fd_<0) throw std::runtime_error("Cannot create frame ring");
        if(ftruncate(fd_,stride_*slots)) { close(fd_); throw std::runtime_error("Cannot size frame ring"); }
        void* p=mmap(nullptr,stride_*slots,PROT_READ|PROT_WRITE,MAP_SHARED,fd_,0);
        if(p==MAP_FAILED) { close(fd_); throw std::runtime_error("Cannot map frame ring"); }
        data_=static_cast<unsigned char*>(p);
    }
    ~FrameRing(){ if(data_)munmap(data_,stride_*slots); if(fd_>=0)close(fd_); }
    size_t write(uint64_t sequence,const cv::Mat& cam0,const cv::Mat& cam1,
                 const cv::Mat& raw=cv::Mat()){
        if(!cam0.isContinuous()||!cam1.isContinuous()||cam0.total()*cam0.elemSize()!=image_bytes||cam1.total()*cam1.elemSize()!=image_bytes)
            throw std::runtime_error("Unexpected frame geometry");
        const size_t slot=sequence%slots;
        if(flock(fd_,LOCK_EX)) throw std::runtime_error("Cannot lock frame ring");
        auto p=data_+slot*stride_;
        std::memcpy(p+8,cam0.data,image_bytes);
        std::memcpy(p+8+image_bytes,cam1.data,image_bytes);
        // One publication and one lock for both BGR eyes and their original
        // RAW. Missing RAW is explicit; stale bytes must never be reused.
        if(with_raw_) {
            const uint64_t valid=raw.type()==CV_8UC2 && raw.rows==1024 && raw.cols==1280;
            std::memcpy(p+slot_bytes,&valid,8);
            if(valid) for(int y=0;y<raw.rows;++y)
                std::memcpy(p+slot_bytes+8+size_t(y)*1280*2,raw.ptr(y),1280*2);
        }
        const uint64_t token=sequence+1;
        std::memcpy(p,&token,8);
        flock(fd_,LOCK_UN);
        return slot;
    }
};
