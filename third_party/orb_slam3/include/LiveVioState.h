#ifndef EGO2_LIVE_VIO_STATE_H
#define EGO2_LIVE_VIO_STATE_H
#include <sophus/se3.hpp>
#include <utility>
namespace ORB_SLAM3 {
// Copied on the tracking thread while the map update lock is held.
struct LiveVioState {
    double timestamp = -1.;
    bool valid = false, imu_initialized = false, initialization_complete = false;
    int tracking_state = 0;
    unsigned long revision = 0;
    std::pair<unsigned long, unsigned long> identity{0, 0};
    Sophus::SE3f camera_from_world;
};
}
#endif
