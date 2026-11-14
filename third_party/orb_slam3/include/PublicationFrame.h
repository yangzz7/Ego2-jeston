// Stable output coordinates for a live map. Does not modify SLAM's map or poses.
#ifndef ORB_SLAM3_PUBLICATION_FRAME_H
#define ORB_SLAM3_PUBLICATION_FRAME_H

#include <sophus/se3.hpp>

namespace ORB_SLAM3 {

class PublicationFrame {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void reset() { ready_ = false; }
    bool ready() const { return ready_; }

    // The reference is a fixed keyframe, never the previous moving camera pose.
    // Freeze the gauge only after inertial initialization, retaining Z-up.
    Sophus::SE3f convert(const Sophus::SE3f& camera_from_world,
                        const Sophus::SE3f& world_from_reference, double world_scale) {
        if (!ready_) {
            base_ = world_from_reference.cast<double>();
            base_scale_ = world_scale;
            ready_ = true;
        }
        const auto reference = world_from_reference.cast<double>();
        const auto camera = camera_from_world.inverse().cast<double>();
        const double scale = base_scale_ / world_scale;
        const Eigen::Matrix3d rotation = base_.rotationMatrix() * reference.rotationMatrix().transpose();
        const Eigen::Vector3d position = base_.translation() + scale * rotation *
            (camera.translation() - reference.translation());
        const Sophus::SE3d stable_camera(rotation * camera.rotationMatrix(), position);
        return stable_camera.inverse().cast<float>();
    }

private:
    bool ready_ = false;
    double base_scale_ = 1.;
    Sophus::SE3d base_;
};

} // namespace ORB_SLAM3
#endif
