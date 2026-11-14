#pragma once

#include <opencv2/core.hpp>
#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

namespace camera_tuner {

enum class PipelineMode { Tunable = 0, Minimal = 1, SdkFast = 2 };
enum class ToneMode { SRgb = 0, Power = 1, Linear = 2 };

struct EyeParameters {
    bool autoWhiteBalance = true;
    // R, G, B multipliers, also used for manual/locked white balance.
    std::array<double, 3> whiteBalance{{1.0, 1.0, 1.0}};
    double blackLevel = 16.0;       // RAW8 DN, subtracted once before demosaic
    double exposureEv = 0.0;       // digital compensation, never sensor exposure
    ToneMode tone = ToneMode::SRgb;
    double gamma = 2.2;
    double blackPoint = 6.0;        // encoded 8-bit domain
    double contrast = 0.3;
    double saturation = 1.0;
    double bayerDenoise = 0.5;      // normalized 0..1
    double lumaDenoise = 0.25;
    double chromaDenoise = 0.5;
    double falseColor = 0.5;
    double sharpening = 0.0;
    bool ccmEnabled = false;
    // R-major matrix applied to linear [R,G,B]. Identity until calibrated.
    std::array<double, 9> ccm{{1,0,0, 0,1,0, 0,0,1}};
};

struct Parameters {
    int schemaVersion = 1;
    std::string serial = "s200032";
    PipelineMode mode = PipelineMode::Tunable;
    bool linked = true;
    std::array<EyeParameters, 2> eyes;
    std::uint64_t revision = 1;
};

struct Telemetry {
    bool valid = false;
    double exposureMs = 0.0;  // FPGA AEC target, not register readback
    double gain = 0.0;
    double temperatureC = 0.0;
};

struct RawStereoFrame {
    // C1 = transport low byte; C2 = high byte. Owning, unmodified RAW8,
    // including the metadata row. Physical left/right must be verified.
    std::array<cv::Mat, 2> raw;
    std::array<Telemetry, 2> telemetry;
    std::string serial;
    int hardwareVersion = 2;
    int firmwareVersion = 6;
    std::uint64_t sequence = 0;
    double deviceTimestamp = 0.0;
    std::chrono::steady_clock::time_point received;
    bool demo = false;
};

struct EyeMetrics {
    std::array<double, 3> whiteBalance{{1,1,1}};
    std::array<int, 256> histogram{};
    double clippedPercent = 0.0;
    double sharpness = 0.0;
};

struct ProcessedFrame {
    std::shared_ptr<const RawStereoFrame> source;
    Parameters parameters;
    std::array<cv::Mat, 2> bgr;
    std::array<EyeMetrics, 2> metrics;
    double processingMs = 0.0;
};

// Throws std::invalid_argument for invalid, nonfinite or unsupported values.
void validateParameters(const Parameters& parameters);
Parameters loadParameters(const std::string& filename);
void saveParameters(const std::string& filename, const Parameters& parameters);

// GUI snapshot/file bundle writing is intentionally outside the portable ISP.

}  // namespace camera_tuner
