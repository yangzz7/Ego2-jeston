#include "tunable_isp.h"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {
using namespace camera_tuner;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void near(double value, double expected, double tolerance, const char* message) {
    if (std::abs(value - expected) > tolerance) {
        std::cerr << message << ": got " << value << ", expected " << expected << '\n';
        throw std::runtime_error(message);
    }
}

// Explicit physical fixture, independent of SelectBayerConversion: firmware
// 03 has B-G/G-R for both eyes; 04/05/06 flip C1 to R-G/G-B only.
std::shared_ptr<RawStereoFrame> uniform(int firmware, int r, int g, int b) {
    auto frame = std::make_shared<RawStereoFrame>();
    frame->firmwareVersion = firmware;
    frame->serial = "synthetic";
    frame->sequence = 7;
    for (int eye = 0; eye < 2; ++eye) {
        frame->raw[eye].create(48, 64, CV_8UC1);
        const bool redFirst = firmware != 3 && eye == 0;
        for (int y = 0; y < frame->raw[eye].rows; ++y) {
            auto* row = frame->raw[eye].ptr<unsigned char>(y);
            for (int x = 0; x < frame->raw[eye].cols; ++x) {
                if ((x & 1) != (y & 1)) row[x] = static_cast<unsigned char>(g);
                else row[x] = static_cast<unsigned char>(((x & 1) == 0) == redFirst ? r : b);
            }
        }
        // This is metadata, not image content.
        frame->raw[eye].row(47).setTo(eye ? 255 : 0);
    }
    return frame;
}

Parameters linear() {
    Parameters parameters;
    for (auto& eye : parameters.eyes) {
        eye.autoWhiteBalance = false;
        eye.tone = ToneMode::Linear;
        eye.blackPoint = 0;
        eye.contrast = 0;
        eye.bayerDenoise = 0;
        eye.lumaDenoise = 0;
        eye.chromaDenoise = 0;
        eye.falseColor = 0;
    }
    return parameters;
}

cv::Scalar interior(const cv::Mat& image) {
    return cv::mean(image(cv::Rect(8, 8, image.cols - 16, image.rows - 16)));
}

void testFirmwarePhaseAndMinimal() {
    TunableIsp isp;
    auto parameters = linear();
    parameters.mode = PipelineMode::Minimal;
    // Make it observable if this mode accidentally applies the tunable knobs.
    parameters.eyes[0].whiteBalance = {{2, 1, 0.7}};
    parameters.eyes[0].exposureEv = 2;
    for (int firmware : {3, 4, 5, 6}) {
        const auto frame = isp.process(uniform(firmware, 180, 80, 35), parameters);
        for (int eye = 0; eye < 2; ++eye) {
            const auto color = interior(frame.bgr[eye]);
            near(color[0], 35, 1, "Minimal blue channel / firmware phase");
            near(color[1], 80, 1, "Minimal green channel / firmware phase");
            near(color[2], 180, 1, "Minimal red channel / firmware phase");
        }
    }
}

void testRawPreservationAndMetadataIsolation() {
    const auto source = uniform(6, 110, 95, 75);
    const auto original0 = source->raw[0].clone();
    const auto original1 = source->raw[1].clone();
    auto alternative = uniform(6, 110, 95, 75);
    alternative->raw[0].row(47).setTo(255);
    alternative->raw[1].row(47).setTo(0);
    for (auto mode : {PipelineMode::Minimal, PipelineMode::Tunable, PipelineMode::SdkFast}) {
        TunableIsp first, second;
        auto parameters = linear();
        parameters.mode = mode;
        const auto a = first.process(source, parameters);
        const auto b = second.process(alternative, parameters);
        for (int eye = 0; eye < 2; ++eye)
            require(cv::norm(a.bgr[eye], b.bgr[eye], cv::NORM_INF) == 0,
                    "Metadata bytes contaminated the displayed image");
        require(a.source.get() == source.get(), "Original source ownership was not retained");
    }
    require(cv::norm(source->raw[0], original0, cv::NORM_INF) == 0 &&
            cv::norm(source->raw[1], original1, cv::NORM_INF) == 0,
            "ISP modified the source RAW/metadata");
}

void testBlackLevelWhiteBalanceCcmAndSaturation() {
    TunableIsp isp;
    const auto raw = uniform(6, 96, 80, 48);
    auto parameters = linear();
    auto result = isp.process(raw, parameters);
    const auto baseline = interior(result.bgr[0]);
    near(baseline[2], (96 - 16) * 255.0 / 239, 1.0, "RAW black level must be subtracted exactly once");
    near(baseline[0], (48 - 16) * 255.0 / 239, 1.0, "Linear full-scale normalization");

    parameters.eyes[0].whiteBalance = {{2, 1, 1}};
    ++parameters.revision;
    result = isp.process(raw, parameters);
    const auto balanced = interior(result.bgr[0]);
    near(balanced[2], (96 - 16) * 2 * 255.0 / 239, 1.0, "Manual red WB applied in RAW domain");
    near(balanced[0], baseline[0], 1.0, "Manual red WB must preserve blue");
    near(result.metrics[0].whiteBalance[0], 2, 0, "Metrics report applied WB");
    near(interior(result.bgr[1])[2], baseline[2], 1.0, "Per-eye state must be independent");
    require(result.parameters.revision == parameters.revision, "Frame lost parameter revision");

    parameters.eyes[0].ccmEnabled = true;
    parameters.eyes[0].ccm = {{0,0,1, 0,1,0, 1,0,0}};
    result = isp.process(raw, parameters);
    const auto swapped = interior(result.bgr[0]);
    near(swapped[0], balanced[2], 1.0, "R-major CCM must map red to BGR blue");
    near(swapped[2], balanced[0], 1.0, "R-major CCM must map blue to BGR red");

    parameters.eyes[0].saturation = 0;
    result = isp.process(raw, parameters);
    const auto gray = interior(result.bgr[0]);
    near(gray[0], gray[1], 1.0, "Zero saturation must produce neutral gray");
    near(gray[1], gray[2], 1.0, "Zero saturation must produce neutral gray");
    near(gray[0], .114 * swapped[0] + .587 * swapped[1] + .299 * swapped[2], 2.0,
         "Saturation must follow CCM and preserve linear luminance");
}

void testHotToneAndExposure() {
    TunableIsp isp;
    const auto raw = uniform(6, 64, 64, 64);
    auto parameters = linear();
    const auto baseline = interior(isp.process(raw, parameters).bgr[0]);
    parameters.eyes[0].exposureEv = 1;
    auto output = interior(isp.process(raw, parameters).bgr[0]);
    near(output[0], baseline[0] * 2, 1.0, "EV must update on the same RAW in linear space");
    parameters.eyes[0].exposureEv = 0;
    parameters.eyes[0].tone = ToneMode::SRgb;
    output = interior(isp.process(raw, parameters).bgr[0]);
    const double normalized = (64 - 16) / 239.0;
    near(output[0], (1.055 * std::pow(normalized, 1.0 / 2.4) - .055) * 255, 1.0,
         "sRGB transfer must occur exactly once");
    parameters.eyes[0].tone = ToneMode::Power;
    parameters.eyes[0].gamma = 2;
    const auto power = interior(isp.process(raw, parameters).bgr[0]);
    near(power[0], std::sqrt(normalized) * 255, 1.0, "Power gamma must hot-update");
    parameters.eyes[0].blackPoint = 20;
    const auto blackPoint = interior(isp.process(raw, parameters).bgr[0]);
    require(blackPoint[0] < power[0], "Encoded black point must hot-update");
    parameters.eyes[0].contrast = 1;
    require(interior(isp.process(raw, parameters).bgr[0])[0] < blackPoint[0],
            "Encoded contrast must hot-update");
    parameters.eyes[0] = linear().eyes[0];
    require(cv::norm(isp.process(raw, parameters).bgr[0],
                     isp.process(raw, linear()).bgr[0], cv::NORM_INF) == 0,
            "Returning controls to baseline must reproduce the same RAW result");
}

void testAutoWhiteBalanceAndMetrics() {
    TunableIsp isp;
    auto parameters = linear();
    parameters.eyes[0].autoWhiteBalance = true;
    const auto result = isp.process(uniform(6, 60, 90, 120), parameters);
    near(result.metrics[0].whiteBalance[0], 74.0 / 44, .01, "AWB red ratio must exclude pedestal");
    near(result.metrics[0].whiteBalance[2], 74.0 / 104, .01, "AWB blue ratio must exclude pedestal");
    const auto color = interior(result.bgr[0]);
    near(color[0], color[1], 2, "AWB should neutralize a uniform gray target");
    near(color[2], color[1], 2, "AWB should neutralize a uniform gray target");
    require(std::accumulate(result.metrics[0].histogram.begin(),
                            result.metrics[0].histogram.end(), 0) == 64 * 47,
            "Histogram must omit metadata row");
    near(result.metrics[0].clippedPercent, 0, 0, "Unsaturated fixture has no clipped pixels");
    near(result.metrics[0].sharpness, 0, 0.1, "Uniform fixture has no sharpness energy");
    parameters = linear();
    const auto clipped = isp.process(uniform(6, 255, 255, 255), parameters);
    near(clipped.metrics[0].clippedPercent, 100, 0, "Clipping metric counts output pixels");
}

void testDenoiseAndSdkMode() {
    auto raw = uniform(6, 90, 90, 90);
    cv::RNG random(731);
    for (auto& eye : raw->raw)
        for (int y = 0; y < eye.rows - 1; ++y)
            for (int x = 0; x < eye.cols; ++x)
                eye.at<unsigned char>(y, x) = static_cast<unsigned char>(90 + random.uniform(-3, 4));
    TunableIsp isp;
    auto parameters = linear();
    const auto original = isp.process(raw, parameters);
    for (auto& eye : parameters.eyes) {
        eye.bayerDenoise = 1;
        eye.lumaDenoise = 1;
        eye.chromaDenoise = 1;
        eye.falseColor = 1;
    }
    const auto filtered = isp.process(raw, parameters);
    require(filtered.metrics[0].sharpness < original.metrics[0].sharpness,
            "NR controls must measurably reduce flat-field noise");

    const auto stable = uniform(6, 100, 90, 80);
    parameters.mode = PipelineMode::SdkFast;
    const auto before = isp.process(stable, parameters);
    parameters.eyes[0].exposureEv = 2;
    parameters.eyes[0].ccmEnabled = true;
    parameters.eyes[0].saturation = 0;
    const auto after = isp.process(stable, parameters);
    require(cv::norm(before.bgr[0], after.bgr[0], cv::NORM_INF) == 0,
            "SDK comparison must ignore tunable parameters");
    require(std::isnan(after.metrics[0].whiteBalance[0]),
            "Unavailable SDK AWB readback must not be presented as 1x");
}

}  // namespace

int main() {
    try {
        cv::setNumThreads(1);
        testFirmwarePhaseAndMinimal();
        testRawPreservationAndMetadataIsolation();
        testBlackLevelWhiteBalanceCcmAndSaturation();
        testHotToneAndExposure();
        testAutoWhiteBalanceAndMetrics();
        testDenoiseAndSdkMode();
        std::cout << "ISP synthetic tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ISP test failure: " << error.what() << '\n';
        return 1;
    }
}
