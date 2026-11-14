#include <cyperstereo_isp/processor.h>
#include <camera_tuner/tunable_isp.h>
#include "smartsens_metadata.h"

#include <limits>
#include <mutex>
#include <stdexcept>

namespace cyperstereo_isp {

std::shared_ptr<RawStereoFrame> decodeTransport(
    const std::uint8_t* bytes, std::size_t byteCount, std::size_t stride,
    const std::string& serial, std::uint64_t sequence) {
    if (serial.empty() || serial.size() > 128 ||
        serial.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-") != std::string::npos)
        throw std::invalid_argument("Invalid camera serial");
    if (!bytes || stride < kTransportRowBytes ||
        stride > (std::numeric_limits<std::size_t>::max() - kTransportRowBytes) / (kHeight - 1) ||
        byteCount < (kHeight - 1) * stride + kTransportRowBytes)
        throw std::invalid_argument("Incomplete 1280x1024 packed stereo transport or invalid row stride");
    const auto* metadata = bytes + (kHeight - 1) * stride;
    const auto word = [metadata](int column) {
        return std::uint16_t(metadata[column * 2]) | (std::uint16_t(metadata[column * 2 + 1]) << 8);
    };
    const int hardware = word(0), firmware = word(1);
    if (!cyperstereo::IsSupportedSmartSensFirmware(hardware, firmware))
        throw std::invalid_argument("Invalid/unsupported SmartSens metadata " + std::to_string(hardware) +
                                    "/" + std::to_string(firmware) + "; expected hardware 2, firmware 3..7");
    if (word(3) >= 10000 || word(4) >= 43200)
        throw std::invalid_argument("Corrupted device timestamp in stereo metadata");
    auto result = std::make_shared<RawStereoFrame>();
    result->serial = serial;
    result->sequence = sequence;
    result->received = std::chrono::steady_clock::now();
    result->hardwareVersion = hardware;
    result->firmwareVersion = firmware;
    result->deviceTimestamp = word(2) * 43200.0 + word(4) + word(3) / 10000.0;
    for (auto& plane : result->raw) plane.create(kHeight, kWidth, CV_8UC1);
    for (int y = 0; y < kHeight; ++y) {
        const auto* row = bytes + y * stride;
        auto* c1 = result->raw[0].ptr<std::uint8_t>(y);
        auto* c2 = result->raw[1].ptr<std::uint8_t>(y);
        for (int x = 0; x < kWidth; ++x) {
            c1[x] = row[2 * x];
            c2[x] = row[2 * x + 1];
        }
    }
    const auto layout = cyperstereo::GetSmartSensMetadataLayout(hardware, firmware);
    if (layout.has_ae_telemetry) {
        for (int eye = 0; eye < 2; ++eye) {
            auto& t = result->telemetry[eye];
            const int lines = word(layout.exposure_base_col + eye);
            const int temperature = word(layout.temperature_base_col + eye);
            const int gain = word(layout.gain_base_col + eye);
            t.exposureMs = lines * layout.line_time_sec * 1000.0;
            t.temperatureC = ((temperature >> 8) * 8 + (temperature & 7)) / 4.0 - 273.15;
            t.gain = gain / 64.0;
            t.valid = lines > 0 && gain > 0 && t.temperatureC >= -60.0 && t.temperatureC <= 150.0;
        }
    }
    return result;
}

struct Processor::Impl {
    mutable std::mutex mutex;
    Parameters parameters;
    camera_tuner::TunableIsp isp;
    std::uint64_t sequence = 0;
    explicit Impl(const Parameters& p) : parameters(p) {}
};

Processor::Processor(const Parameters& parameters) {
    validateParameters(parameters);
    impl_ = std::make_unique<Impl>(parameters);
}
Processor::~Processor() = default;

Parameters Processor::parameters() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->parameters;
}
void Processor::setParameters(const Parameters& parameters) {
    validateParameters(parameters);
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->parameters = parameters;
}
void Processor::loadConfig(const std::string& path) { setParameters(loadParameters(path)); }
void Processor::setDisplayBackend(bool cuda) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (cuda && impl_->parameters.mode != PipelineMode::Tunable)
        throw std::invalid_argument("CUDA display supports the tunable pipeline; use CPU for comparison modes");
    impl_->isp.setDisplayBackend(cuda);
}
void Processor::reset() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->isp.reset();
}
ProcessedFrame Processor::processTransport(const std::uint8_t* bytes, std::size_t byteCount,
                                          std::size_t stride) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto frame = decodeTransport(bytes, byteCount, stride, impl_->parameters.serial, impl_->sequence + 1);
    auto result = impl_->isp.process(std::move(frame), impl_->parameters);
    ++impl_->sequence;
    return result;
}
ProcessedFrame Processor::processRaw(const RawStereoFrame& frame) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto owned = std::make_shared<RawStereoFrame>(frame);
    for (int eye = 0; eye < 2; ++eye) owned->raw[eye] = frame.raw[eye].clone();
    return impl_->isp.process(std::move(owned), impl_->parameters);
}

}  // namespace cyperstereo_isp
