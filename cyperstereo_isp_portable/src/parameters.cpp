#include "camera_tuner/model.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <system_error>
#include <cmath>
#include <set>
#include <stdexcept>

namespace camera_tuner {
namespace {

[[noreturn]] void invalid(const std::string& field, const std::string& reason) {
    throw std::invalid_argument(field + ": " + reason);
}

void range(double value, double low, double high, const std::string& field) {
    if (!std::isfinite(value) || value < low || value > high)
        invalid(field, "must be finite and in [" + std::to_string(low) + ", " + std::to_string(high) + "]");
}

void keys(const cv::FileNode& node, const std::set<std::string>& expected, const std::string& name) {
    if (!node.isMap()) invalid(name, "expected a mapping");
    std::set<std::string> actual;
    for (auto it = node.begin(); it != node.end(); ++it) actual.insert((*it).name());
    if (actual != expected || node.size() != expected.size()) invalid(name, "missing, duplicate, or unknown fields");
}

int integer(const cv::FileNode& node, const std::string& name) {
    if (!node.isInt()) invalid(name, "expected an integer");
    return static_cast<int>(node);
}

bool boolean(const cv::FileNode& node, const std::string& name) {
    const int value = integer(node, name);
    if (value != 0 && value != 1) invalid(name, "expected integer 0 or 1");
    return value == 1;
}

double number(const cv::FileNode& node, const std::string& name) {
    if (!node.isInt() && !node.isReal()) invalid(name, "expected a number");
    const double value = static_cast<double>(node);
    if (!std::isfinite(value)) invalid(name, "must be finite");
    return value;
}

std::string string(const cv::FileNode& node, const std::string& name) {
    if (!node.isString()) invalid(name, "expected a string");
    return static_cast<std::string>(node);
}

template<std::size_t N>
void readArray(const cv::FileNode& node, std::array<double, N>& out, const std::string& name) {
    if (!node.isSeq() || node.size() != N) invalid(name, "incorrect array length");
    for (std::size_t i = 0; i < N; ++i) out[i] = number(node[static_cast<int>(i)], name);
}

const char* modeName(PipelineMode mode) {
    switch (mode) {
        case PipelineMode::Tunable: return "tunable";
        case PipelineMode::Minimal: return "minimal";
        case PipelineMode::SdkFast: return "sdk_fast";
    }
    invalid("mode", "unsupported pipeline mode");
}

const char* toneName(ToneMode tone) {
    switch (tone) {
        case ToneMode::SRgb: return "srgb";
        case ToneMode::Power: return "power";
        case ToneMode::Linear: return "linear";
    }
    invalid("tone", "unsupported tone mode");
}

EyeParameters readEye(const cv::FileNode& node) {
    keys(node, {"auto_white_balance", "white_balance", "black_level", "exposure_ev", "tone", "gamma",
                "black_point", "contrast", "saturation", "bayer_denoise", "luma_denoise", "chroma_denoise",
                "false_color", "sharpening", "ccm_enabled", "ccm"}, "eye");
    EyeParameters eye;
    eye.autoWhiteBalance = boolean(node["auto_white_balance"], "auto_white_balance");
    readArray(node["white_balance"], eye.whiteBalance, "white_balance");
    eye.blackLevel = number(node["black_level"], "black_level");
    eye.exposureEv = number(node["exposure_ev"], "exposure_ev");
    const auto tone = string(node["tone"], "tone");
    if (tone == "srgb") eye.tone = ToneMode::SRgb;
    else if (tone == "power") eye.tone = ToneMode::Power;
    else if (tone == "linear") eye.tone = ToneMode::Linear;
    else invalid("tone", "expected srgb, power or linear");
    eye.gamma = number(node["gamma"], "gamma");
    eye.blackPoint = number(node["black_point"], "black_point");
    eye.contrast = number(node["contrast"], "contrast");
    eye.saturation = number(node["saturation"], "saturation");
    eye.bayerDenoise = number(node["bayer_denoise"], "bayer_denoise");
    eye.lumaDenoise = number(node["luma_denoise"], "luma_denoise");
    eye.chromaDenoise = number(node["chroma_denoise"], "chroma_denoise");
    eye.falseColor = number(node["false_color"], "false_color");
    eye.sharpening = number(node["sharpening"], "sharpening");
    eye.ccmEnabled = boolean(node["ccm_enabled"], "ccm_enabled");
    readArray(node["ccm"], eye.ccm, "ccm");
    return eye;
}

void writeEye(cv::FileStorage& out, const EyeParameters& e) {
    out << "{" << "auto_white_balance" << static_cast<int>(e.autoWhiteBalance);
    out << "white_balance" << "[";
    for (double value : e.whiteBalance) out << value;
    out << "]" << "black_level" << e.blackLevel << "exposure_ev" << e.exposureEv;
    out << "tone" << toneName(e.tone) << "gamma" << e.gamma << "black_point" << e.blackPoint;
    out << "contrast" << e.contrast << "saturation" << e.saturation;
    out << "bayer_denoise" << e.bayerDenoise << "luma_denoise" << e.lumaDenoise;
    out << "chroma_denoise" << e.chromaDenoise << "false_color" << e.falseColor;
    out << "sharpening" << e.sharpening << "ccm_enabled" << static_cast<int>(e.ccmEnabled);
    out << "ccm" << "[";
    for (double value : e.ccm) out << value;
    out << "]" << "}";
}

}  // namespace

void validateParameters(const Parameters& p) {
    if (p.schemaVersion != 1) invalid("schema_version", "only version 1 is supported");
    if (p.serial.empty() || p.serial.size() > 128 ||
        p.serial.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-") != std::string::npos)
        invalid("serial", "expected a nonempty camera serial (letters, numbers, underscore, hyphen)");
    modeName(p.mode);
    for (const auto& e : p.eyes) {
        toneName(e.tone);
        for (double value : e.whiteBalance) range(value, .25, 4., "white_balance");
        range(e.blackLevel, 0, 64, "black_level");
        range(e.exposureEv, -3, 3, "exposure_ev");
        range(e.gamma, 1, 4, "gamma");
        range(e.blackPoint, 0, 32, "black_point");
        range(e.contrast, 0, 1, "contrast");
        range(e.saturation, 0, 2, "saturation");
        range(e.bayerDenoise, 0, 1, "bayer_denoise");
        range(e.lumaDenoise, 0, 1, "luma_denoise");
        range(e.chromaDenoise, 0, 1, "chroma_denoise");
        range(e.falseColor, 0, 1, "false_color");
        range(e.sharpening, 0, 1, "sharpening");
        for (double value : e.ccm) range(value, -4, 4, "ccm");
    }
}

Parameters loadParameters(const std::string& filename) {
    if (filename.empty() || filename.find('\0') != std::string::npos)
        invalid("filename", "expected a nonempty path without NUL bytes");
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot read parameters: " + filename);
    // Bound the bytes actually read, including a file that grows while open.
    std::string bytes;
    std::array<char, 4096> buffer{};
    while (file) {
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = static_cast<std::size_t>(file.gcount());
        if (bytes.size() + count > 1024 * 1024)
            invalid("configuration", "file exceeds 1 MiB");
        bytes.append(buffer.data(), count);
    }
    if (file.bad() || !file.eof()) throw std::runtime_error("Cannot finish reading parameters: " + filename);
    try {
        cv::FileStorage input(bytes, cv::FileStorage::READ | cv::FileStorage::MEMORY);
        if (!input.isOpened()) invalid("configuration", "cannot parse YAML");
        keys(input.root(), {"schema_version", "serial", "mode", "linked", "eyes"}, "configuration");
        Parameters p;
        p.schemaVersion = integer(input["schema_version"], "schema_version");
        p.serial = string(input["serial"], "serial");
        const auto mode = string(input["mode"], "mode");
        if (mode == "tunable") p.mode = PipelineMode::Tunable;
        else if (mode == "minimal") p.mode = PipelineMode::Minimal;
        else if (mode == "sdk_fast") p.mode = PipelineMode::SdkFast;
        else invalid("mode", "expected tunable, minimal or sdk_fast");
        p.linked = boolean(input["linked"], "linked");
        auto eyes = input["eyes"];
        if (!eyes.isSeq() || eyes.size() != 2) invalid("eyes", "expected exactly two eyes");
        p.eyes[0] = readEye(eyes[0]);
        p.eyes[1] = readEye(eyes[1]);
        p.revision = 1;
        validateParameters(p);
        return p;
    } catch (const cv::Exception& e) {
        invalid("configuration", e.what());
    }
}

void saveParameters(const std::string& filename, const Parameters& p) {
    validateParameters(p);
    cv::FileStorage output("parameters.yaml", cv::FileStorage::WRITE | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_YAML);
    output << "schema_version" << p.schemaVersion << "serial" << p.serial;
    output << "mode" << modeName(p.mode) << "linked" << static_cast<int>(p.linked) << "eyes" << "[";
    for (const auto& eye : p.eyes) writeEye(output, eye);
    output << "]";
    const auto yaml = output.releaseAndGetString();
    if (filename.empty() || filename.find('\0') != std::string::npos)
        invalid("filename", "expected a nonempty path without NUL bytes");
    const std::filesystem::path target(filename);
    if (target.filename().empty()) invalid("filename", "expected a file path");
    const auto parent = target.has_parent_path() ? target.parent_path() : std::filesystem::path(".");
    // fopen's standard C11 x mode exclusively creates a new file. Checking
    // exists() before an ordinary ofstream open would race other writers and
    // could truncate a pre-existing temporary file. No direct-write fallback.
    static std::atomic<unsigned long long> sequence{0};
    std::random_device entropy;
    const auto seed = static_cast<unsigned long long>(entropy()) ^
        static_cast<unsigned long long>(std::chrono::steady_clock::now().time_since_epoch().count());
    struct TemporaryFile {
        std::filesystem::path path;
        std::FILE* stream = nullptr;
        bool owned = false;
        ~TemporaryFile() {
            if (stream) std::fclose(stream);
            if (owned) { std::error_code ignored; std::filesystem::remove(path, ignored); }
        }
    } temporary;
    for (unsigned attempt = 0; attempt < 64; ++attempt) {
        temporary.path = parent / ("." + target.filename().string() + ".tmp-" +
            std::to_string(seed) + "-" + std::to_string(sequence.fetch_add(1)));
        temporary.stream = std::fopen(temporary.path.string().c_str(), "wbx");
        if (temporary.stream) { temporary.owned = true; break; }
        if (errno != EEXIST)
            throw std::system_error(errno, std::generic_category(), "Cannot create temporary parameters file");
    }
    if (!temporary.stream) throw std::runtime_error("Cannot allocate a unique temporary parameters file");
    const std::size_t count = std::fwrite(yaml.data(), 1, yaml.size(), temporary.stream);
    if (count != yaml.size() || std::fflush(temporary.stream) != 0)
        throw std::runtime_error("Cannot finish writing temporary parameters file: " + temporary.path.string());
    std::FILE* stream = temporary.stream;
    temporary.stream = nullptr;
    if (std::fclose(stream) != 0)
        throw std::runtime_error("Cannot close temporary parameters file: " + temporary.path.string());
    std::error_code error;
    std::filesystem::rename(temporary.path, target, error);
    if (error) throw std::system_error(error, "Cannot atomically replace parameters: " + filename);
    temporary.owned = false;
    // rename gives atomic replacement on the target filesystem. This API does
    // not promise power-loss durability (which would require platform fsync).
}

}  // namespace camera_tuner
