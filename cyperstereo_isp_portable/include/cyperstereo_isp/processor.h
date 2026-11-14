#pragma once

#include <camera_tuner/model.h>
#include <cstddef>
#include <memory>

namespace cyperstereo_isp {

using Parameters = camera_tuner::Parameters;
using RawStereoFrame = camera_tuner::RawStereoFrame;
using ProcessedFrame = camera_tuner::ProcessedFrame;
using PipelineMode = camera_tuner::PipelineMode;
using ToneMode = camera_tuner::ToneMode;
using camera_tuner::loadParameters;
using camera_tuner::saveParameters;
using camera_tuner::validateParameters;

inline constexpr int kWidth = 1280;
inline constexpr int kHeight = 1024;
inline constexpr std::size_t kTransportRowBytes = kWidth * 2;

// Pure memory decoder: no camera, Qt, ROS, Rerun, or hardware writes.
// Input is full native S2 transport, labelled YUYV but actually alternating
// C1/C2 RAW8 bytes. Preserve its final metadata row. Stride may include padding.
// This function copies both planes and decodes firmware/gain/exposure metadata.
std::shared_ptr<RawStereoFrame> decodeTransport(
    const std::uint8_t* bytes, std::size_t byteCount, std::size_t stride,
    const std::string& serial = "s200032", std::uint64_t sequence = 0);

// Keep one instance per stereo camera across frames to retain AWB history.
// All operations on one instance are serialized. Returned Mats own their data
// and remain valid across later calls. Output order is C1 then C2, BGR8.
class Processor {
public:
    explicit Processor(const Parameters& parameters = Parameters{});
    ~Processor();
    Processor(const Processor&) = delete;
    Processor& operator=(const Processor&) = delete;

    Parameters parameters() const;
    void setParameters(const Parameters& parameters);
    void loadConfig(const std::string& path);
    void reset();
    void setDisplayBackend(bool cuda);
    ProcessedFrame processTransport(const std::uint8_t* bytes, std::size_t byteCount,
                                    std::size_t stride = kTransportRowBytes);
    // Advanced input: full uncropped RAW8 planes, including metadata last row.
    // hardwareVersion/firmwareVersion must be accurate; provide valid telemetry
    // for gain-adaptive denoising. Missing telemetry uses gain 1. Input is copied.
    ProcessedFrame processRaw(const RawStereoFrame& frame);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cyperstereo_isp
