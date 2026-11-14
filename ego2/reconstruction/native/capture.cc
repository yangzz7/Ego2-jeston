#include "System.h"
#include "Thirdparty/usb/uvc/cyperstereo_api.h"
#include <opencv2/imgcodecs.hpp>
#include <nlohmann/json.hpp>
#include <zmq.h>
#include <csignal>
#include <deque>
#include <fstream>
#include <set>
#include <cstdio>
#include "imu_interval.h"
#include "display_raw_ring.h"
#include "../../common/native/cyper_cuda_loader.h"
#include <cstdlib>
#ifdef EGO2_MULTI_VIO
#include "../../multi_vio/native/frame_ring.h"
#include "../../multi_vio/native/archive.h"
#endif

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
static volatile std::sig_atomic_t interrupted = 0;
static void on_signal(int) { interrupted = 1; }

struct Frame {
    cv::Mat left, right; // cam1, cam0; Bayer initially, then BGR
    cv::Mat packed; // CUDA path owns original C1/C2 bytes until conversion.
    double stamp = 0, device_stamp = 0, host_stamp = 0;
    uint64_t sequence = 0;
    uint32_t firmware = 0;
};
struct Queue {
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<Frame> frames;
    std::deque<ORB_SLAM3::IMU::Point> imu;
    double last_image = -1, last_imu = -1, next_image = -1;
    uint64_t sequence = 0, dropped = 0, samples = 0;
    std::string error;
};

static void atomic_json(const std::string& path, const json& data) {
    std::ofstream out(path + ".tmp");
    out << data.dump() << '\n';
    out.close();
    if (!out || std::rename((path + ".tmp").c_str(), path.c_str()))
        throw std::runtime_error("Cannot write " + path);
}

static void white_balance(cv::Mat& image) {
    // Same bounded gray-world correction as the existing RGB/gesture adapter;
    // no rotation, rectification or recognition overlays enter the raw views.
    cv::Mat sample, lut(1,256,CV_8UC3);
    cv::resize(image,sample,cv::Size(160,120),0,0,cv::INTER_AREA);
    const auto mean=cv::mean(sample);
    const double average=(mean[0]+mean[1]+mean[2])/3.0;
    for(int c=0;c<3;++c) {
        const double gain=std::max(0.5,std::min(2.0,average/std::max(1.0,mean[c])));
        for(int i=0;i<256;++i) lut.at<cv::Vec3b>(0,i)[c]=cv::saturate_cast<uchar>(i*gain);
    }
    cv::LUT(image,lut,image);
}

static int run(int argc, char** argv) {
    if (argc < 12 || argc > 14) throw std::runtime_error("Use python3 -m ego2.reconstruction to launch this backend");
    const std::string voc=argv[1], settings=argv[2], session=argv[3], endpoint=argv[4], serial=argv[5];
    const double fps=std::stod(argv[6]), shift=std::stod(argv[7]), scale=std::stod(argv[8]);
    const double max_gap=std::stod(argv[9]), seconds=std::stod(argv[10]);
    const bool stereo_only = std::string(argv[11]) == "stereo";
    cv::setNumThreads(2);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::shared_ptr<cyperstereo::uvc::device> device;
    if (!cyperstereo::FindCyperstereoDevices(device, serial))
        throw std::runtime_error("Camera " + serial + " not found (check device access and other camera processes)");
    const auto actual = cyperstereo::uvc::get_serial_number(*device);
    const auto& profile = cyperstereo::SelectProfile(actual, *device);
    if (actual != serial || profile.num_cameras != 2 || profile.cam_width != 1280 || profile.frame_height != 1024)
        throw std::runtime_error("Device/profile does not match requested camera " + serial + " (1280x1024 stereo required)");
    std::unique_ptr<ego2::CudaCapture> cuda_capture;
    if(ego2::cuda_capture_enabled()) cuda_capture.reset(new ego2::CudaCapture(profile.cam_width,profile.frame_height));
    std::cout << "[capture-convert] " << (cuda_capture?"cuda":"cpu")
              << ": RAW lanes + Bayer to BGR; USB/metadata on CPU" << std::endl;
    ORB_SLAM3::System slam(voc, settings, stereo_only ? ORB_SLAM3::System::STEREO : ORB_SLAM3::System::IMU_STEREO, false);
#ifdef EGO2_MULTI_VIO
    const bool bundled_preview_raw = std::getenv("EGO2_VIO_PREVIEW_RAW") != nullptr;
    FrameRing ring(session+"/frames.bin", bundled_preview_raw);
    ControlledArchive archive(argc >= 13 ? argv[12] : "", argc == 14 ? argv[13] : "");
    slam.EnableStablePublication(true);
#endif
    void* context = zmq_ctx_new();
    void* publisher = zmq_socket(context, ZMQ_PUB);
    int hwm=2, linger=0;
    zmq_setsockopt(publisher, ZMQ_SNDHWM, &hwm, sizeof(hwm));
    zmq_setsockopt(publisher, ZMQ_LINGER, &linger, sizeof(linger));
    if (zmq_bind(publisher, endpoint.c_str())) {
        zmq_close(publisher); zmq_ctx_term(context);
#ifdef EGO2_MULTI_VIO
        archive.finish();
#endif
        slam.FinishReconstruction();
        throw std::runtime_error("Cannot bind local preview transport");
    }
    cyperstereo::FrameInfo info(profile);
    Queue queue;
    std::unique_ptr<DisplayRawRing> display_raw;
    if (const char* path = std::getenv("EGO2_DISPLAY_RAW_RING")) {
        try { display_raw.reset(new DisplayRawRing(path)); }
        catch (const std::exception& e) { std::cerr << "[display-isp] " << e.what() << std::endl; }
    }
    cyperstereo::uvc::set_device_mode(*device, profile.frame_width, profile.frame_height,
        static_cast<int>(cyperstereo::Format::YUYV), profile.fps,
        [&](const void* data, std::function<void()> continuation) {
            try {
                const double received=std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
                // Parse in the USB callback so dropping an RGB pair never drops
                // the IMU packets embedded in that pair's transport buffer.
                cyperstereo::SetStreamData(info, data, continuation, bool(cuda_capture));
                std::lock_guard<std::mutex> frame_lock(info.mtx);
                const auto& fs = info.framestream;
                std::lock_guard<std::mutex> lock(queue.mutex);
                if (!queue.error.empty() || interrupted) return;
                if (!std::isfinite(fs.image_timestamp) || fs.image_timestamp <= 2) return;
                if (fs.image_timestamp == queue.last_image) return; // rejected/stale SDK frame
                if (fs.image_timestamp < queue.last_image)
                    throw std::runtime_error("Device image clock moved backwards; start a new session");
                queue.last_image = fs.image_timestamp;
                for (int i=0; i<fs.imu.imu_count; ++i) {
                    const double t=fs.imu.imu_timestamp[i];
                    if (t <= queue.last_imu) continue; // firmware may repeat boundary samples
                    const double ax=fs.imu.acc_x[i]*scale, ay=fs.imu.acc_y[i]*scale, az=fs.imu.acc_z[i]*scale;
                    const double gx=fs.imu.gyro_x[i], gy=fs.imu.gyro_y[i], gz=fs.imu.gyro_z[i];
                    if (!std::isfinite(t+ax+ay+az+gx+gy+gz) ||
                        ax*ax+ay*ay+az*az > 40000 || gx*gx+gy*gy+gz*gz > 2500)
                        throw std::runtime_error("Invalid IMU sample; refusing to integrate corrupted measurements");
#ifdef EGO2_MULTI_VIO
                    archive.imu(received,t,ax,ay,az,gx,gy,gz);
#endif
                    queue.imu.emplace_back(ax,ay,az,gx,gy,gz,t);
                    queue.last_imu=t; ++queue.samples;
                }
                if (queue.imu.size()>4000) throw std::runtime_error("VIO stalled: IMU queue exceeded 20 seconds");
                if (fs.image_timestamp >= queue.next_image) {
                    Frame f;
                    if(cuda_capture)
                        f.packed=cv::Mat(profile.frame_height,profile.cam_width,CV_8UC2,const_cast<void*>(data)).clone();
                    else { f.left=fs.left_image.clone(); f.right=fs.right_image.clone(); }
                    f.device_stamp=fs.image_timestamp; f.stamp=f.device_stamp+shift;
                    f.host_stamp=received;
                    f.firmware=fs.software_version; f.sequence=queue.sequence++;
                    if (fs.hardware_version != 2 || f.firmware < 3 || f.firmware > 7)
                        throw std::runtime_error("Unsupported Bayer/metadata firmware");
                    queue.frames.push_back(std::move(f));
                    while (queue.frames.size()>2) {queue.frames.pop_front(); ++queue.dropped;}
                    queue.next_image=fs.image_timestamp+1.0/fps-0.003;
                }
                queue.changed.notify_one();
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(queue.mutex);
                queue.error=e.what(); queue.changed.notify_one();
            }
        });
    std::deque<Frame> history;
    std::set<std::string> saved;
    auto snapshot = [&](bool final) {
        json map=json::parse(slam.ReconstructionSnapshot());
        map["final"]=final; map["mode"]=stereo_only ? "stereo" : "stereo-inertial";
        for (auto& m : map["maps"]) for (auto& kf : m["keyframes"]) {
            const std::string key=kf["image_key"];
            if (!saved.count(key)) {
                for (auto& f : history) if (std::abs(f.stamp-kf["timestamp"].get<double>())<1e-6) {
                    const std::string base=session+"/keyframes/"+key;
                    if (!cv::imwrite(base+"_cam1.jpg", f.left, {cv::IMWRITE_JPEG_QUALITY,95}) ||
                        !cv::imwrite(base+"_cam0.jpg", f.right, {cv::IMWRITE_JPEG_QUALITY,95}))
                        throw std::runtime_error("Cannot save keyframe images");
                    saved.insert(key); break;
                }
            }
            kf["image_available"]=saved.count(key)>0;
        }
        {
            std::lock_guard<std::mutex> lock(queue.mutex);
            map["imu_samples"]=queue.samples; map["dropped_rgb"]=queue.dropped;
        }
        atomic_json(session+"/maps.json", map);
    };
    std::string failure;
    bool streaming=false;
    try {
        cyperstereo::uvc::start_streaming(*device, 0); streaming=true;
        const auto start=Clock::now();
        double next_snapshot=-1;
        double last_sent_imu=-1;
        while (!interrupted && (seconds<=0 || std::chrono::duration<double>(Clock::now()-start).count()<seconds)) {
#ifdef EGO2_MULTI_VIO
            archive.poll();
#endif
            Frame frame;
            std::vector<ORB_SLAM3::IMU::Point> measurements;
            bool gap=false;
            {
                std::unique_lock<std::mutex> lock(queue.mutex);
                const auto ready = [&] {
                    return interrupted || !queue.error.empty() || (!queue.frames.empty() &&
                        (stereo_only || (!queue.imu.empty() && queue.imu.back().t >= queue.frames.front().stamp)));
                };
                if (!queue.changed.wait_for(lock, std::chrono::seconds(5), ready))
                    throw std::runtime_error("No stereo frame bracketed by IMU for 5 seconds; check USB/IMU stream");
                if (interrupted) break;
                if (!queue.error.empty()) throw std::runtime_error(queue.error);
                frame=std::move(queue.frames.front()); queue.frames.pop_front();
                if (stereo_only) queue.imu.clear();
                else {
                    const auto interval=ego2::take_interval(queue.imu,frame.stamp,max_gap,measurements);
                    if (interval==ego2::Interval::waiting) continue;
                    gap=interval==ego2::Interval::gap;
                }
            }
            if (gap) {
                std::cerr << "[reconstruction] IMU gap; resetting active map and skipping fusion" << std::endl;
                slam.ResetActiveMap(); history.clear(); continue;
            }
            cv::Mat preview_raw;
#ifdef EGO2_MULTI_VIO
            if(bundled_preview_raw) {
                // CUDA input is immutable; retain its owned storage until
                // publication. CPU Bayer planes need a copy before conversion.
                if(cuda_capture) preview_raw=frame.packed;
                else cv::merge(std::vector<cv::Mat>{frame.right,frame.left},preview_raw);
            }
#endif
            if(cuda_capture) {
                if(display_raw) display_raw->write_packed(frame.sequence,frame.packed);
                cuda_capture->decode(frame.packed,frame.firmware,frame.right,frame.left);
                frame.packed.release();
            } else {
            if (display_raw) display_raw->write(frame.sequence, frame.right, frame.left);
            // Metadata rows are not scene pixels. Preserve coordinates and
            // replace with same Bayer parity before demosaicing.
            frame.left.row(frame.left.rows-3).copyTo(frame.left.row(frame.left.rows-1));
            frame.right.row(frame.right.rows-3).copyTo(frame.right.row(frame.right.rows-1));
            cv::cvtColor(frame.left,frame.left,cv::COLOR_BayerRG2BGR);
            cv::cvtColor(frame.right,frame.right,frame.firmware>=4 ? cv::COLOR_BayerBG2BGR : cv::COLOR_BayerRG2BGR);
            }
            white_balance(frame.left); white_balance(frame.right);
            measurements=ego2::only_new_samples(measurements,last_sent_imu);
            slam.TrackStereo(frame.left,frame.right,frame.stamp,measurements);
#ifdef EGO2_MULTI_VIO
            json meta=json::parse(slam.LiveVioSnapshot(frame.stamp));
            meta["protocol"]=1; meta["serial"]=serial;
            meta["timestamp"]=frame.device_stamp; meta["slam_timestamp"]=frame.stamp;
            meta["host_monotonic"]=frame.host_stamp; meta["frame_id"]=frame.sequence;
            meta["firmware"]=frame.firmware; meta["cameras"]={"cam0","cam1"};
            meta["slot"]=ring.write(frame.sequence,frame.right,frame.left,preview_raw);
            { std::lock_guard<std::mutex> lock(queue.mutex);
              meta["imu_samples"]=queue.samples; meta["dropped_rgb"]=queue.dropped; }
            archive.append(frame.right,frame.left,meta);
            const auto header=meta.dump();
            zmq_send(publisher,header.data(),header.size(),0);
#else
            history.push_back(frame);
            while (history.size()>64) history.pop_front();
            if (frame.stamp>=next_snapshot) { snapshot(false); next_snapshot=frame.stamp+0.2; }
            json meta={{"version",1},{"timestamp",frame.device_stamp},{"slam_timestamp",frame.stamp},
                       {"frame_id",frame.sequence},{"state",slam.GetTrackingState()}, {"cameras",{"cam1","cam0"}}};
            std::vector<uchar> a,b;
            cv::imencode(".jpg",frame.left,a,{cv::IMWRITE_JPEG_QUALITY,90});
            cv::imencode(".jpg",frame.right,b,{cv::IMWRITE_JPEG_QUALITY,90});
            const auto header=meta.dump();
            zmq_send(publisher,header.data(),header.size(),ZMQ_SNDMORE);
            zmq_send(publisher,a.data(),a.size(),ZMQ_SNDMORE);
            zmq_send(publisher,b.data(),b.size(),0);
#endif
        }
    } catch (const std::exception& e) { failure=e.what(); }
    if (streaming) cyperstereo::uvc::stop_streaming(*device);
#ifdef EGO2_MULTI_VIO
    try { archive.finish(); }
    catch (const std::exception& e) { if(failure.empty())failure=e.what(); }
#endif
    slam.FinishReconstruction();
#ifndef EGO2_MULTI_VIO
    snapshot(true);
#endif
    zmq_close(publisher); zmq_ctx_term(context);
    if (!failure.empty()) throw std::runtime_error(failure);
    return 0;
}

int main(int argc, char** argv) {
    try { return run(argc, argv); }
    catch (const std::exception& e) { std::cerr << "[fatal] " << e.what() << std::endl; return 1; }
}
