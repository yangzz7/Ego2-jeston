#include <cyperstereo_isp/c_api.h>
#include <cyperstereo_isp/processor.h>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string error;
using cyperstereo_isp::Processor;
Processor& checked(void* handle) {
    if (!handle) throw std::invalid_argument("ISP handle is null or closed");
    return *static_cast<Processor*>(handle);
}
template<class Work> int run(Work work) noexcept {
    try { error.clear(); work(); return 0; }
    catch (const std::exception& e) { error = e.what(); }
    catch (...) { error = "Unknown ISP error"; }
    return -1;
}
std::size_t outputSize(uint8_t* pointer, size_t bytes, size_t stride) {
    constexpr size_t row = cyperstereo_isp::kWidth * 3;
    constexpr size_t height = cyperstereo_isp::kHeight;
    if (!pointer || stride < row || stride > (std::numeric_limits<size_t>::max() - row) / (height - 1))
        throw std::invalid_argument("Invalid BGR output pointer/stride");
    const auto required = (height - 1) * stride + row;
    if (bytes < required) throw std::invalid_argument("BGR output buffer is too small");
    return required;
}
bool overlaps(const void* a, size_t an, const void* b, size_t bn) {
    const auto av = reinterpret_cast<uintptr_t>(a), bv = reinterpret_cast<uintptr_t>(b);
    return av <= bv ? bv - av < an : av - bv < bn;
}
}

extern "C" {
int cisp_api_version(void) { return 1; }
const char* cisp_last_error(void) { return error.c_str(); }
void* cisp_create(const char* config_path) {
    Processor* result = nullptr;
    run([&] { result = new Processor(config_path ? cyperstereo_isp::loadParameters(config_path)
                                                : cyperstereo_isp::Parameters{}); });
    return result;
}
void cisp_destroy(void* handle) { delete static_cast<Processor*>(handle); }
void* cisp_create_display(const char* config_path, int cuda) {
    Processor* result = nullptr;
    run([&] {
        if (cuda != 0 && cuda != 1) throw std::invalid_argument("Invalid display backend");
        auto owned = std::make_unique<Processor>(config_path ? cyperstereo_isp::loadParameters(config_path)
                                                           : cyperstereo_isp::Parameters{});
        owned->setDisplayBackend(cuda != 0);
        result = owned.release();
    });
    return result;
}
int cisp_load_config(void* handle, const char* path) {
    return run([&] {
        if (!path) throw std::invalid_argument("Configuration path is null");
        checked(handle).loadConfig(path);
    });
}
int cisp_reset(void* handle) { return run([&] { checked(handle).reset(); }); }
int cisp_process_transport(void* handle, const uint8_t* input, size_t input_bytes, size_t input_stride,
                          uint8_t* c1, size_t c1_bytes, size_t c1_stride,
                          uint8_t* c2, size_t c2_bytes, size_t c2_stride, double* wb) {
    return run([&] {
        auto& processor = checked(handle);
        const auto size1 = outputSize(c1, c1_bytes, c1_stride);
        const auto size2 = outputSize(c2, c2_bytes, c2_stride);
        if (overlaps(c1, size1, c2, size2) || overlaps(input, input_bytes, c1, size1) ||
            overlaps(input, input_bytes, c2, size2))
            throw std::invalid_argument("Input and output buffers must not overlap");
        const auto frame = processor.processTransport(input, input_bytes, input_stride);
        for (int y = 0; y < cyperstereo_isp::kHeight; ++y) {
            std::memcpy(c1 + y * c1_stride, frame.bgr[0].ptr(y), cyperstereo_isp::kWidth * 3);
            std::memcpy(c2 + y * c2_stride, frame.bgr[1].ptr(y), cyperstereo_isp::kWidth * 3);
        }
        if (wb) for (int eye = 0; eye < 2; ++eye)
            for (int color = 0; color < 3; ++color) wb[eye * 3 + color] = frame.metrics[eye].whiteBalance[color];
    });
}
}
