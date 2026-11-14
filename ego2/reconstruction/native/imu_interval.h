#pragma once
#include "ImuTypes.h"
#include <deque>
#include <vector>

namespace ego2 {
enum class Interval { waiting, ready, gap };
inline std::vector<ORB_SLAM3::IMU::Point> only_new_samples(
        const std::vector<ORB_SLAM3::IMU::Point>& interval, double& last_sent) {
    // ORB retains its final boundary sample internally. Re-sending it on the
    // next TrackStereo call creates a zero dt and NaN in PreintegrateIMU.
    std::vector<ORB_SLAM3::IMU::Point> result;
    for (const auto& point : interval) if (point.t>last_sent) {
        result.push_back(point); last_sent=point.t;
    }
    return result;
}
// Consume all samples up to an image's IMU-clock time, retaining the boundary
// and future samples for the next image. No measurements are discarded merely
// because intermediate RGB images were skipped.
inline Interval take_interval(std::deque<ORB_SLAM3::IMU::Point>& queue,
                              double stamp, double max_gap,
                              std::vector<ORB_SLAM3::IMU::Point>& out) {
    if (queue.size()<2 || queue.front().t>stamp || queue.back().t<stamp)
        return Interval::waiting;
    bool gap=false;
    while (queue.size()>1 && queue[1].t<=stamp) {
        if (queue[1].t-queue[0].t>max_gap) gap=true;
        out.push_back(queue.front()); queue.pop_front();
    }
    if (queue.front().t==stamp) {
        out.push_back(queue.front());
    } else {
        if (queue.size()<2) return Interval::waiting;
        const auto a=queue[0], b=queue[1];
        if (b.t-a.t>max_gap) gap=true;
        out.push_back(a);
        const float f=float((stamp-a.t)/(b.t-a.t));
        const auto acc=a.a+f*(b.a-a.a), gyro=a.w+f*(b.w-a.w);
        ORB_SLAM3::IMU::Point boundary(acc.x(),acc.y(),acc.z(),gyro.x(),gyro.y(),gyro.z(),stamp);
        out.push_back(boundary);
        queue.front()=boundary;
    }
    return gap ? Interval::gap : Interval::ready;
}
}
