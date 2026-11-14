#include "archive.h"
#include <unistd.h>
#include <cstdio>
#include <stdexcept>
#include <sys/stat.h>
int main() {
    char name[]="/tmp/ego2-archive-test-XXXXXX";
    if(!mkdtemp(name))return 1;
    const std::string dir=name;
    try {
        NativeArchive archive(dir);
        archive.imu(3.0,0,0,9.8,1,2,3);
        const cv::Mat a(16,24,CV_8UC3,cv::Scalar(1,2,3)),b(16,24,CV_8UC3,cv::Scalar(4,5,6));
        archive.append(a,b,{{"frame_id",7},{"map_id","a"}});
        archive.finish();
        const auto left=cv::imread(dir+"/7_cam1.png"),right=cv::imread(dir+"/7_cam0.png");
        if(cv::norm(left,b)!=0||cv::norm(right,a)!=0)throw std::runtime_error("eye swap or pixel change");
        nlohmann::json frame,imu;
        std::ifstream frames(dir+"/frames.jsonl"),samples(dir+"/imu.jsonl");
        frames>>frame;samples>>imu;
        if(frame["frame_id"]!=7||imu["timestamp_seconds"]!=3.)throw std::runtime_error("metadata mismatch");
        mkdir((dir+"/native").c_str(),0700);
        ControlledArchive controlled("",dir);
        auto command=[&](const nlohmann::json& value){std::ofstream out(dir+"/command.json");out<<value.dump();};
        command({{"id","one"},{"phase","prepare"},{"directory",dir}});controlled.poll();
        controlled.append(a,b,{{"frame_id",1},{"host_monotonic",99.}});
        command({{"id","one"},{"phase","commit"},{"start_time",100.}});controlled.poll();
        controlled.append(a,b,{{"frame_id",2},{"host_monotonic",99.9}});
        controlled.imu(99.9,3.0,0,0,9.8,1,2,3);
        controlled.append(a,b,{{"frame_id",3},{"host_monotonic",100.1}});
        controlled.imu(100.1,3.1,0,0,9.8,1,2,3);
        auto stop=[&](const std::string& id,double start,double end) {
            command({{"id",id},{"phase","stop"},{"start_time",start},{"stop_time",end}});
            for(int i=0;i<1000;++i) {
                controlled.poll();
                nlohmann::json state;std::ifstream input(dir+"/state.json");input>>state;
                if(state.value("id","")==id&&state.value("stopped",false))return;
                usleep(1000);
            }
            throw std::runtime_error("stop not acknowledged");
        };
        stop("one",100.,101.);
        controlled.append(a,b,{{"frame_id",4},{"host_monotonic",102.}});
        controlled.imu(102.,3.2,0,0,9.8,1,2,3);
        std::ifstream recorded(dir+"/native/frames.jsonl"),recorded_imu(dir+"/native/imu.jsonl");
        recorded>>frame;recorded_imu>>imu;
        if(frame["frame_id"]!=3||imu["timestamp_seconds"]!=3.1)throw std::runtime_error("pre-record data leaked");
        std::string trailing;if(recorded>>trailing)throw std::runtime_error("unexpected archive frame");
        if(recorded_imu>>trailing)throw std::runtime_error("post-stop IMU leaked");
        mkdir((dir+"/second").c_str(),0700);mkdir((dir+"/second/native").c_str(),0700);
        command({{"id","two"},{"phase","prepare"},{"directory",dir+"/second"}});controlled.poll();
        command({{"id","two"},{"phase","commit"},{"start_time",200.}});controlled.poll();
        controlled.append(a,b,{{"frame_id",5},{"host_monotonic",200.1}});
        stop("two",200.,201.);
        std::ifstream second(dir+"/second/native/frames.jsonl");second>>frame;
        if(frame["frame_id"]!=5||second>>trailing)throw std::runtime_error("second recording not isolated");
        for(const auto& file:{"5_cam0.png","5_cam1.png","frames.jsonl","imu.jsonl"})
            std::remove((dir+"/second/native/"+file).c_str());
        rmdir((dir+"/second/native").c_str());rmdir((dir+"/second").c_str());
        for(const auto& file:{"3_cam0.png","3_cam1.png","frames.jsonl","imu.jsonl"})
            std::remove((dir+"/native/"+file).c_str());
        rmdir((dir+"/native").c_str());
        for(const auto& file:{"command.json","state.json"})std::remove((dir+"/"+file).c_str());
    } catch(...) {return 2;}
    for(const auto& file:{"7_cam0.png","7_cam1.png","frames.jsonl","imu.jsonl"})
        std::remove((dir+"/"+file).c_str());
    rmdir(dir.c_str());
    return 0;
}
