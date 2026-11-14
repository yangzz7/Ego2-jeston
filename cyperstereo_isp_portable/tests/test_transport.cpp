#include <cyperstereo_isp/processor.h>
#include <opencv2/core.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool good, const char* message) { if (!good) throw std::runtime_error(message); }
template<class F> void rejects(F fn) {
    try { fn(); } catch (const std::exception&) { return; }
    throw std::runtime_error("Malformed input was accepted");
}
}
int main() {
    try {
        using namespace cyperstereo_isp;
        constexpr std::size_t stride = kTransportRowBytes + 17;
        std::vector<std::uint8_t> packed(kHeight * stride, 0xab);
        for (int y = 0; y < kHeight; ++y) for (int x = 0; x < kWidth; ++x) {
            packed[y * stride + 2 * x] = 70;
            packed[y * stride + 2 * x + 1] = 120;
        }
        auto word = [&](int column, int value) {
            const auto offset = (kHeight - 1) * stride + column * 2;
            packed[offset] = value & 255; packed[offset + 1] = (value >> 8) & 255;
        };
        word(0, 2); word(1, 6); word(2, 1); word(3, 5000); word(4, 10);
        word(69, 168); word(70, 168); word(73, 149 * 256 + 1); word(74, 149 * 256 + 1);
        word(77, 128); word(78, 192);
        const auto original = packed;
        auto frame = decodeTransport(packed.data(), packed.size(), stride, "s200032", 17);
        require(frame->sequence == 17 && frame->firmwareVersion == 6, "Identity/firmware");
        require(frame->raw[0].at<unsigned char>(100, 100) == 70 &&
                frame->raw[1].at<unsigned char>(100, 100) == 120, "Transport lane/stride");
        require(frame->raw[0].at<unsigned char>(1023, 0) == 2, "Original metadata was changed");
        require(std::abs(frame->deviceTimestamp - 43210.5) < 1e-8, "Timestamp");
        require(frame->telemetry[0].valid && frame->telemetry[1].valid &&
                frame->telemetry[0].gain == 2 && frame->telemetry[1].gain == 3, "Gain metadata");
        require(std::abs(frame->telemetry[0].exposureMs - 168 * 452.0 / 15187500.0 * 1000) < 1e-8,
                "Firmware 6 exposure line time");
        auto parameters = Parameters{}; parameters.mode = PipelineMode::Minimal;
        Processor isp(parameters);
        auto output = isp.processTransport(packed.data(), packed.size(), stride);
        const auto preserved = output.bgr[0].clone();
        word(0, 3);
        rejects([&] { isp.processTransport(packed.data(), packed.size(), stride); });
        word(0, 2);
        auto second = isp.processTransport(packed.data(), packed.size(), stride);
        require(cv::norm(preserved, output.bgr[0], cv::NORM_INF) == 0, "Output lifetime");
        require(second.source->sequence == 2, "Failed process advanced sequence");
        require(packed == original, "Input modified");
        rejects([&] { decodeTransport(nullptr, packed.size(), stride); });
        rejects([&] { decodeTransport(packed.data(), 20, stride); });
        rejects([&] { decodeTransport(packed.data(), packed.size(), 1); });
        rejects([&] { decodeTransport(packed.data(), packed.size(), std::numeric_limits<size_t>::max()); });
        word(3, 10000); rejects([&] { decodeTransport(packed.data(), packed.size(), stride); });
        word(3, 5000); word(1, 3);
        require(!decodeTransport(packed.data(), packed.size(), stride)->telemetry[0].valid, "Legacy telemetry fabricated");
        std::cout << "Transport and ownership checks passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
