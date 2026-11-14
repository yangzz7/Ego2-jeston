#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include "Thirdparty/usb/uvc/cyperstereo_api.h"
#include "../../common/native/cyper_cuda_loader.h"
#include "display_raw_ring.h"

int main(int argc,char** argv) {
    cv::setNumThreads(1);
    ego2::CudaCapture decoder(1280,1024);
    for(int fw=3;fw<=7;++fw) {
        cv::Mat raw(1024,1280,CV_8UC2);cv::randu(raw,0,256);
        if(argc==2) {
            std::ifstream in(argv[1],std::ios::binary);
            in.read(reinterpret_cast<char*>(raw.data),raw.total()*2);
            assert(in.gcount()==std::streamsize(raw.total()*2));
            fw=raw.at<cv::Vec2b>(1023,1)[0];
        } else {
            raw.row(1023).setTo(cv::Scalar(0,0));
            auto word=[&](int col,int value){raw.at<cv::Vec2b>(1023,col)=cv::Vec2b(value&255,value>>8);};
            word(0,2);word(1,fw);word(4,12);
            for(int i=0;i<(fw==5?13:7);++i) {
                int base=5+9*i;word(base,1000+50*i);word(base+1,12);
                word(base+2,100);word(base+3,65500);word(base+4,5000);
                word(base+5,10);word(base+6,65530);word(base+7,30);word(base+8,16);
            }
        }
        const auto original=raw.clone();
        cyperstereo::FrameInfo cpu(cyperstereo::kProfileSmartSensDuo),gpu(cyperstereo::kProfileSmartSensDuo);
        gpu.framestream.left_image.setTo(23);gpu.framestream.right_image.setTo(42);
        cyperstereo::SetStreamData(cpu,raw.data,[]{});
        cyperstereo::SetStreamData(gpu,raw.data,[]{},true);
        const auto& a=cpu.framestream;const auto& b=gpu.framestream;
        assert(a.image_timestamp==b.image_timestamp && a.software_version==b.software_version);
        assert(a.hardware_version==b.hardware_version && a.imu.imu_count==b.imu.imu_count);
        for(int i=0;i<a.imu.imu_count;++i) {
            assert(a.imu.acc_x[i]==b.imu.acc_x[i] && a.imu.acc_y[i]==b.imu.acc_y[i] && a.imu.acc_z[i]==b.imu.acc_z[i]);
            assert(a.imu.gyro_x[i]==b.imu.gyro_x[i] && a.imu.gyro_y[i]==b.imu.gyro_y[i] && a.imu.gyro_z[i]==b.imu.gyro_z[i]);
            assert(a.imu.temperature[i]==b.imu.temperature[i] && a.imu.imu_timestamp[i]==b.imu.imu_timestamp[i]);
        }
        for(int i=0;i<4;++i) {
            assert(a.exposure_lines[i]==b.exposure_lines[i] && a.exposure_time[i]==b.exposure_time[i]);
            assert(a.camera_temperature[i]==b.camera_temperature[i] && a.camera_gain[i]==b.camera_gain[i]);
            assert(a.image_midpoint_timestamp[i]==b.image_midpoint_timestamp[i]);
        }
        assert(cv::countNonZero(b.left_image.row(0)!=23)==0); // No full image split in SDK.
        assert(cv::countNonZero(b.right_image.row(0)!=42)==0);
        cv::Mat ca=a.right_image.clone(),cb=a.left_image.clone(),ga,gb;
        ca.row(1021).copyTo(ca.row(1023));cb.row(1021).copyTo(cb.row(1023));
        cv::cvtColor(ca,ca,fw>=4?cv::COLOR_BayerBG2BGR:cv::COLOR_BayerRG2BGR);
        cv::cvtColor(cb,cb,cv::COLOR_BayerRG2BGR);
        decoder.decode(raw,fw,ga,gb);
        assert(cv::norm(ca,ga,cv::NORM_INF)==0 && cv::norm(cb,gb,cv::NORM_INF)==0);
        assert(cv::norm(raw,original,cv::NORM_INF)==0);
        char dir[]="/tmp/ego2-cuda-ring-XXXXXX";assert(mkdtemp(dir));
        std::string p=std::string(dir)+"/raw";
        {
            DisplayRawRing ring(p);assert(ring.write_packed(7,raw));
            int fd=open(p.c_str(),O_RDONLY);assert(fd>=0 && !flock(fd,LOCK_SH));
            assert(!ring.write_packed(8,raw));flock(fd,LOCK_UN);close(fd);
            std::ifstream in(p,std::ios::binary);in.seekg(DisplayRawRing::slot_bytes);
            uint64_t token;in.read(reinterpret_cast<char*>(&token),8);assert(token==8);
            cv::Mat check(raw.size(),raw.type());in.read(reinterpret_cast<char*>(check.data),check.total()*2);
            assert(cv::norm(check,raw,cv::NORM_INF)==0);
        }
        unlink(p.c_str());rmdir(dir);
        // Corrupt markers must keep the previous timestamp and reject publication.
        const auto before=gpu.framestream.image_timestamp;
        raw.at<cv::Vec2b>(1023,0)[0]=99;
        cyperstereo::SetStreamData(gpu,raw.data,[]{},true);
        assert(gpu.framestream.image_timestamp==before && gpu.meta_bad_count==1);
        if(argc==2)break;
    }
    std::cout << "PASS: native CUDA pixels, metadata, corrupt frame rejection, packed display ring" << std::endl;
}
