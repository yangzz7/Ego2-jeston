#pragma once
#include <sys/file.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <string>
#include <opencv2/core.hpp>

// Optional display-only side channel. Original Bayer planes (including the
// metadata row) are reinterleaved as C1=cam0, C2=cam1. A busy reader drops this
// display copy rather than blocking SLAM or the lossless BGR archive.
class DisplayRawRing {
    int fd_ = -1;
    unsigned char* data_ = nullptr;
public:
    static constexpr size_t payload_bytes = 1280u*1024u*2u;
    static constexpr size_t slot_bytes = 8+payload_bytes;
    static constexpr size_t slots = 3;
    explicit DisplayRawRing(const std::string& path) {
        fd_ = open(path.c_str(), O_RDWR|O_CREAT|O_EXCL|O_CLOEXEC, 0600);
        if(fd_ < 0) throw std::runtime_error("Cannot create display RAW ring");
        if(ftruncate(fd_, slots*slot_bytes)) {
            close(fd_); throw std::runtime_error("Cannot size display RAW ring");
        }
        void* mapped=mmap(nullptr,slots*slot_bytes,PROT_READ|PROT_WRITE,MAP_SHARED,fd_,0);
        if(mapped==MAP_FAILED) { close(fd_); throw std::runtime_error("Cannot map display RAW ring"); }
        data_=static_cast<unsigned char*>(mapped);
    }
    ~DisplayRawRing() { if(data_)munmap(data_,slots*slot_bytes); if(fd_>=0)close(fd_); }
    bool write_packed(uint64_t sequence, const cv::Mat& raw) {
        if(raw.type()!=CV_8UC2 || raw.rows!=1024 || raw.cols!=1280) return false;
        if(flock(fd_,LOCK_EX|LOCK_NB)) return false;
        auto* slot=data_+(sequence%slots)*slot_bytes;
        for(int y=0;y<raw.rows;++y)
            std::memcpy(slot+8+size_t(y)*1280*2,raw.ptr(y),1280*2);
        const uint64_t token=sequence+1;
        std::memcpy(slot,&token,8);
        flock(fd_,LOCK_UN);
        return true;
    }
    bool write(uint64_t sequence, const cv::Mat& c1, const cv::Mat& c2) {
        if(c1.type()!=CV_8UC1 || c2.type()!=CV_8UC1 ||
           c1.rows!=1024 || c1.cols!=1280 || c2.size()!=c1.size()) return false;
        if(flock(fd_,LOCK_EX|LOCK_NB)) return false;
        auto* slot=data_+(sequence%slots)*slot_bytes;
        for(int y=0;y<1024;++y) {
            auto* dst=slot+8+y*1280*2;
            const auto* a=c1.ptr<unsigned char>(y);
            const auto* b=c2.ptr<unsigned char>(y);
            for(int x=0;x<1280;++x) { dst[2*x]=a[x]; dst[2*x+1]=b[x]; }
        }
        const uint64_t token=sequence+1;
        std::memcpy(slot,&token,8);
        flock(fd_,LOCK_UN);
        return true;
    }
};
