#include <cyperstereo_isp/processor.h>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
std::vector<std::uint8_t> readRaw(const char* path) {
    constexpr std::size_t size = cyperstereo_isp::kWidth * cyperstereo_isp::kHeight;
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != static_cast<std::streamoff>(size))
        throw std::runtime_error(std::string("Expected 1280*1024 RAW8 bytes: ") + path);
    input.seekg(0);
    std::vector<std::uint8_t> data(size);
    if (!input.read(reinterpret_cast<char*>(data.data()), size)) throw std::runtime_error("RAW read failed");
    return data;
}
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: isp_convert CONFIG.yaml C1.raw8 C2.raw8 NEW_OUTPUT_DIRECTORY\n"
                     "RAW files must be full native planes including the original metadata row.\n";
        return 2;
    }
    try {
        const auto a = readRaw(argv[2]), b = readRaw(argv[3]);
        std::vector<std::uint8_t> packed(a.size() * 2);
        for (std::size_t i = 0; i < a.size(); ++i) { packed[2 * i] = a[i]; packed[2 * i + 1] = b[i]; }
        cyperstereo_isp::Processor isp(cyperstereo_isp::loadParameters(argv[1]));
        const auto frame = isp.processTransport(packed.data(), packed.size());
        const std::filesystem::path directory(argv[4]);
        if (!std::filesystem::create_directory(directory)) throw std::runtime_error("Output directory already exists");
        for (int eye = 0; eye < 2; ++eye) {
            const auto path = directory / ("c" + std::to_string(eye + 1) + "_color.png");
            if (!cv::imwrite(path.string(), frame.bgr[eye])) throw std::runtime_error("Cannot write PNG");
            const auto& wb = frame.metrics[eye].whiteBalance;
            std::cout << "C" << eye + 1 << " WB RGB: " << wb[0] << ", " << wb[1] << ", " << wb[2] << '\n';
        }
        cyperstereo_isp::saveParameters((directory / "parameters.yaml").string(), frame.parameters);
        std::cout << "Firmware " << frame.source->hardwareVersion << '/' << frame.source->firmwareVersion
                  << ", processing " << frame.processingMs << " ms; " << directory << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
