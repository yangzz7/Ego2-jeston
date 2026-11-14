#pragma once
#include <condition_variable>
#include <deque>
#include <fstream>
#include <mutex>
#include <thread>
#include <memory>
#include <limits>
#include <cstdio>
#include <future>
#include <chrono>
#include <opencv2/imgcodecs.hpp>
#include <nlohmann/json.hpp>

// An archive branch before the lossy UI transport. Bounded overload is an error,
// never a silent dropped archive frame. The Python supervisor retains its cache.
class NativeArchive {
    struct Item { cv::Mat cam0,cam1; nlohmann::json meta; };
    std::string directory_,error_;
    std::ofstream frames_,imu_;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<Item> queue_;
    std::thread thread_;
    bool stopping_=false;
    void work() {
        try {
            for (;;) {
                Item item;
                { std::unique_lock<std::mutex> lock(mutex_);
                  changed_.wait(lock,[&]{return stopping_||!queue_.empty();});
                  if(queue_.empty())break;
                  item=std::move(queue_.front());queue_.pop_front(); }
                const auto id=std::to_string(item.meta["frame_id"].get<uint64_t>());
                for(int cam=0;cam<2;++cam) {
                    const auto name=id+"_cam"+std::to_string(cam)+".png";
                    if(!cv::imwrite(directory_+"/"+name,cam==0?item.cam0:item.cam1,
                                   {cv::IMWRITE_PNG_COMPRESSION,1}))
                        throw std::runtime_error("Cannot save trajectory image");
                    item.meta["images"]["cam"+std::to_string(cam)]=name;
                }
                frames_<<item.meta.dump()<<'\n';frames_.flush();
                if(!frames_)throw std::runtime_error("Cannot save trajectory frame journal");
            }
        } catch(const std::exception& e) {
            std::lock_guard<std::mutex> lock(mutex_);error_=e.what();
        }
    }
public:
    explicit NativeArchive(const std::string& directory):directory_(directory) {
        if(directory_.empty())return;
        frames_.open(directory_+"/frames.jsonl");imu_.open(directory_+"/imu.jsonl");
        if(!frames_||!imu_)throw std::runtime_error("Cannot open trajectory archive");
        thread_=std::thread([this]{work();});
    }
    bool enabled()const{return !directory_.empty();}
    void imu(double t,double ax,double ay,double az,double gx,double gy,double gz) {
        if(!enabled())return;
        imu_<<nlohmann::json{{"timestamp_seconds",t},{"acceleration_m_s2",{ax,ay,az}},
                            {"angular_velocity_rad_s",{gx,gy,gz}}}.dump()<<'\n';
        if(!imu_)throw std::runtime_error("Cannot save trajectory IMU");
    }
    void append(const cv::Mat& cam0,const cv::Mat& cam1,const nlohmann::json& meta) {
        if(!enabled())return;
        std::lock_guard<std::mutex> lock(mutex_);
        if(!error_.empty())throw std::runtime_error(error_);
        if(queue_.size()>=16)throw std::runtime_error("Trajectory archive cannot keep up; stopping without silent loss");
        queue_.push_back({cam0,cam1,meta});changed_.notify_one();
    }
    void finish() {
        if(!enabled()||!thread_.joinable())return;
        {std::lock_guard<std::mutex> lock(mutex_);stopping_=true;changed_.notify_one();}
        thread_.join();imu_.flush();frames_.flush();
        if(!error_.empty())throw std::runtime_error(error_);
        if(!imu_||!frames_)throw std::runtime_error("Trajectory archive flush failed");
    }
    ~NativeArchive(){try{finish();}catch(...) {}}
};

// Control is polled on the VIO thread; a short mutex protects activation from
// the USB callback. Preparation opens files without recording any samples.
class ControlledArchive {
    std::string control_,seen_,token_;
    std::unique_ptr<NativeArchive> archive_;
    std::mutex mutex_;
    double start_=std::numeric_limits<double>::infinity();
    double end_=std::numeric_limits<double>::infinity();
    std::future<void> finishing_;
    void reply(const nlohmann::json& value) {
        const auto path=control_+"/state.json";
        std::ofstream stream(path+".tmp");stream<<value.dump()<<'\n';stream.close();
        if(!stream||std::rename((path+".tmp").c_str(),path.c_str()))
            throw std::runtime_error("Cannot acknowledge recording preparation");
    }
public:
    ControlledArchive(const std::string& initial,const std::string& control):control_(control) {
        if(!initial.empty()){archive_.reset(new NativeArchive(initial));start_=-std::numeric_limits<double>::infinity();}
    }
    void poll() {
        if(control_.empty())return;
        std::lock_guard<std::mutex> lock(mutex_);
        std::ifstream stream(control_+"/command.json");
        if(stream) {
            nlohmann::json command;stream>>command;
            const auto id=command["id"].get<std::string>(),phase=command["phase"].get<std::string>();
            if(seen_!=id+phase) {
                seen_=id+phase;
                if(phase=="prepare") {
                    start_=end_=std::numeric_limits<double>::infinity();
                    try {
                        if(finishing_.valid())finishing_.get();
                        if(archive_)archive_->finish();archive_.reset();
                        archive_.reset(new NativeArchive(command["directory"].get<std::string>()+"/native"));
                        token_=id;reply({{"id",id},{"ready",true}});
                    } catch(const std::exception& e) {reply({{"id",id},{"ready",false},{"error",e.what()}});}
                } else if(phase=="commit"&&id==token_&&archive_)start_=command["start_time"].get<double>();
                else if(phase=="stop"&&id==token_) {
                    start_=command.value("start_time",start_);
                    end_=command["stop_time"].get<double>();
                } else if(phase=="abort") {
                    start_=std::numeric_limits<double>::infinity();
                    if(archive_)archive_->finish();archive_.reset();
                }
            }
        }
        const double now=std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        if(archive_&&now>=end_) {
            // Move the writer out before draining it, keeping USB and VIO free.
            auto detached=std::move(archive_);
            finishing_=std::async(std::launch::async,[writer=std::move(detached)]{writer->finish();});
        }
        if(finishing_.valid()&&finishing_.wait_for(std::chrono::seconds(0))==std::future_status::ready) {
            try {finishing_.get();reply({{"id",token_},{"stopped",true}});}
            catch(const std::exception& e){reply({{"id",token_},{"stopped",false},{"error",e.what()}});}
        }
    }
    void imu(double received,double t,double ax,double ay,double az,double gx,double gy,double gz) {
        std::lock_guard<std::mutex> lock(mutex_);
        if(archive_&&received>=start_&&received<end_)archive_->imu(t,ax,ay,az,gx,gy,gz);
    }
    void append(const cv::Mat& a,const cv::Mat& b,const nlohmann::json& meta) {
        std::lock_guard<std::mutex> lock(mutex_);
        const double received=meta["host_monotonic"].get<double>();
        if(archive_&&received>=start_&&received<end_)archive_->append(a,b,meta);
    }
    void finish(){std::lock_guard<std::mutex> lock(mutex_);if(archive_)archive_->finish();if(finishing_.valid())finishing_.get();}
};
