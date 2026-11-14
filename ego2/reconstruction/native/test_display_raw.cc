#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <fstream>
#include "display_raw_ring.h"

int main(int argc, char** argv) {
    char directory[]="/tmp/ego2-display-ring-XXXXXX";
    assert(mkdtemp(directory));
    const std::string path=argc>1 ? argv[1] : std::string(directory)+"/raw.bin";
    cv::Mat a(1024,1280,CV_8UC1,cv::Scalar(31)), b(1024,1280,CV_8UC1,cv::Scalar(97));
    a.at<unsigned char>(1023,0)=2;
    a.at<unsigned char>(1023,1)=7;
    a.at<unsigned char>(1023,1279)=123;
    const auto before_a=a.clone(),before_b=b.clone();
    {
        DisplayRawRing ring(path);
        assert(ring.write(4,a,b));
        int fd=open(path.c_str(),O_RDONLY);
        assert(fd>=0 && !flock(fd,LOCK_SH));
        assert(!ring.write(5,a,b)); // Capture never waits for the viewer.
        flock(fd,LOCK_UN); close(fd);
        assert(cv::norm(a,before_a,cv::NORM_INF)==0 && cv::norm(b,before_b,cv::NORM_INF)==0);
        std::ifstream in(path,std::ios::binary);
        const size_t offset=DisplayRawRing::slot_bytes;
        in.seekg(offset);
        uint64_t token; in.read(reinterpret_cast<char*>(&token),8); assert(token==5);
        unsigned char bytes[2]; in.read(reinterpret_cast<char*>(bytes),2);
        assert(bytes[0]==31 && bytes[1]==97);
        in.seekg(offset+8+1023*1280*2);
        in.read(reinterpret_cast<char*>(bytes),2); assert(bytes[0]==2 && bytes[1]==97);
        in.seekg(offset+8+DisplayRawRing::payload_bytes-2);
        in.read(reinterpret_cast<char*>(bytes),2); assert(bytes[0]==123 && bytes[1]==97);
        assert(!ring.write(6,cv::Mat(),b));
    }
    if(argc==1) unlink(path.c_str());
    rmdir(directory);
}
