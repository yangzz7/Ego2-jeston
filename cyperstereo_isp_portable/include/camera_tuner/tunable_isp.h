#pragma once

#include "model.h"
#include <memory>

namespace camera_tuner {

// Owns both eyes' processing buffers and AWB histories. Calls are serialized;
// the capture thread must hand over owning RAW frames, never its UVC buffer.
class TunableIsp {
public:
    TunableIsp();
    ~TunableIsp();
    TunableIsp(const TunableIsp&) = delete;
    TunableIsp& operator=(const TunableIsp&) = delete;

    ProcessedFrame process(std::shared_ptr<const RawStereoFrame> source,
                           const Parameters& parameters);
    void reset();
    // Display-only acceleration; default callers retain the original CPU path.
    void setDisplayBackend(bool cuda);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace camera_tuner
