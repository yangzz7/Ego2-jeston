#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <fstream>
#include <string>
#include "frame_ring.h"
#include "PublicationFrame.h"
#include "Thirdparty/usb/uvc/smartsens_metadata.h"

int main(){
    using namespace cyperstereo;
    assert(IsSupportedSmartSensFirmware(2,7));
    assert(!IsSupportedSmartSensFirmware(2,9));
    const auto layout=GetSmartSensMetadataLayout(2,7);
    assert(layout.imu_samples_per_frame==7 && layout.ae_marker_col==68 && layout.end_col==106);
    assert(layout.line_time_sec==GetSmartSensMetadataLayout(2,6).line_time_sec);
    char directory[]="/tmp/ego2-ring-test-XXXXXX";
    assert(mkdtemp(directory));
    const std::string path=std::string(directory)+"/frames.bin";
    {
        FrameRing ring(path);
        cv::Mat a(1024,1280,CV_8UC3,cv::Scalar(20,30,40)),b(1024,1280,CV_8UC3,cv::Scalar(90,100,110));
        assert(ring.write(4,a,b)==1);
        std::ifstream in(path,std::ios::binary); in.seekg(FrameRing::slot_bytes);
        uint64_t token=0; in.read(reinterpret_cast<char*>(&token),8); assert(token==5);
        unsigned char pixel[3]; in.read(reinterpret_cast<char*>(pixel),3); assert(pixel[0]==20&&pixel[2]==40);
        in.seekg(FrameRing::slot_bytes+8+FrameRing::image_bytes); in.read(reinterpret_cast<char*>(pixel),3); assert(pixel[0]==90);
    }
    unlink(path.c_str()); rmdir(directory);
    ORB_SLAM3::PublicationFrame frame;
    const Sophus::SE3f ref(Sophus::SO3f(),Eigen::Vector3f(1,0,0));
    const Sophus::SE3f camera(Sophus::SO3f(),Eigen::Vector3f(2,1,0));
    const auto first=frame.convert(camera.inverse(),ref,1.);
    const Sophus::SE3f correction(Sophus::SO3f::exp(Eigen::Vector3f(0,0,.4f)),Eigen::Vector3f(3,2,1));
    const auto next=frame.convert((correction*camera).inverse(),correction*ref,1.);
    assert((first.matrix()-next.matrix()).norm()<1e-5);
}
