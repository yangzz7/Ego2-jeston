#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include "frame_ring.h"

// Also used by Python to test the actual C++/Python wire format, not a mock.
int main(int argc,char** argv) {
    assert(argc==3);
    FrameRing ring(argv[1],true);
    const std::string mode=argv[2];
    auto write=[&](uint64_t n,bool present) {
        const int value=n%199;
        cv::Mat a(1024,1280,CV_8UC3,cv::Scalar(value,value+1,value+2));
        cv::Mat b(1024,1280,CV_8UC3,cv::Scalar(value+3,value+4,value+5));
        // Non-contiguous RAW catches stride mistakes; include metadata row.
        cv::Mat storage(1024,1282,CV_8UC2,cv::Scalar(value+6,value+7));
        cv::Mat raw=storage(cv::Rect(1,0,1280,1024));
        raw.at<cv::Vec2b>(1023,0)=cv::Vec2b(2,7);
        auto aa=a.clone(),bb=b.clone(),rr=raw.clone();
        ring.write(n,a,b,present?raw:cv::Mat());
        assert(cv::norm(a,aa,cv::NORM_INF)==0);
        assert(cv::norm(b,bb,cv::NORM_INF)==0);
        assert(cv::norm(raw,rr,cv::NORM_INF)==0);
        std::cout<<n<<std::endl;
    };
    if(mode=="stress") {
        for(uint64_t n=1;n<302;n+=3) {
            write(n,true);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } else {
        write(1,true);
        write(4,mode!="missing");
    }
}
