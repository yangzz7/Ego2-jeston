// Local storage implementation for the SDK's unchanged ::Frame/ImageMem ABI.
// camera_tuner links this translation unit INSTEAD OF src/ISP/.../frame.cpp.
// Keep the SDK's pointer layout, element types, and padding so its ISP stages
// can swap buffers normally, but never delete[] a void pointer. Every pointer
// and buffer starts at zero, and partial construction releases all allocations.
#include "common/frame.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {

template<class T>
void releaseArray(void*& pointer) noexcept {
    delete[] static_cast<T*>(pointer);
    pointer = nullptr;
}

void releaseStorage(ImageMem& memory) noexcept {
    releaseArray<std::uint8_t>(memory.raw_u8_i);
    releaseArray<std::uint16_t>(memory.raw_u16_i);
    releaseArray<std::uint16_t>(memory.raw_u16_o);
    releaseArray<std::int32_t>(memory.raw_s32_i);
    releaseArray<std::int32_t>(memory.raw_s32_o);
    releaseArray<std::int32_t>(memory.bgr_s32_i);
    releaseArray<std::int32_t>(memory.bgr_s32_o);
    releaseArray<std::uint8_t>(memory.bgr_u8_o);
    releaseArray<float>(memory.yuv_f32_i.y);
    releaseArray<float>(memory.yuv_f32_i.u);
    releaseArray<float>(memory.yuv_f32_i.v);
    releaseArray<float>(memory.yuv_f32_o.y);
    releaseArray<float>(memory.yuv_f32_o.u);
    releaseArray<float>(memory.yuv_f32_o.v);
    releaseArray<std::uint8_t>(memory.yuv_u8_i.y);
    releaseArray<std::uint8_t>(memory.yuv_u8_i.u);
    releaseArray<std::uint8_t>(memory.yuv_u8_i.v);
    releaseArray<std::uint8_t>(memory.yuv_u8_o.y);
    releaseArray<std::uint8_t>(memory.yuv_u8_o.u);
    releaseArray<std::uint8_t>(memory.yuv_u8_o.v);
}

template<class T>
void allocateArray(void*& destination, std::size_t count) {
    destination = new T[count]{};
}

std::size_t paddedPixels(int width, int height) {
    if (width <= 0 || height <= 0)
        throw std::invalid_argument("ISP frame dimensions must be positive");
    const auto paddedWidth = static_cast<std::size_t>(width) + 8;
    const auto paddedHeight = static_cast<std::size_t>(height) + 8;
    const auto maximum = std::numeric_limits<std::size_t>::max();
    if (paddedWidth > maximum / paddedHeight)
        throw std::length_error("ISP frame dimensions overflow size_t");
    const auto pixels = paddedWidth * paddedHeight;
    // The largest individual allocation is a three-channel int32 BGR plane.
    if (pixels > maximum / (3 * sizeof(std::int32_t)))
        throw std::length_error("ISP frame storage size overflows size_t");
    return pixels;
}

bool validRawLength(const ImageInfo& info, int length) {
    return length >= 0 && static_cast<std::size_t>(length) <=
        static_cast<std::size_t>(info.width) * static_cast<std::size_t>(info.height) * 2;
}

}  // namespace

Frame::Frame(ImageInfo& imageInfo) : info(imageInfo), data{} {
    FrameDateMalloc(info.width, info.height);
}

int Frame::FrameDateMalloc(int width, int height) {
    const auto pixels = paddedPixels(width, height);
    try {
        allocateArray<std::uint8_t>(data.raw_u8_i, pixels * 2);
        allocateArray<std::uint16_t>(data.raw_u16_i, pixels);
        allocateArray<std::uint16_t>(data.raw_u16_o, pixels);
        allocateArray<std::int32_t>(data.raw_s32_i, pixels);
        allocateArray<std::int32_t>(data.raw_s32_o, pixels);
        allocateArray<std::int32_t>(data.bgr_s32_i, pixels * 3);
        allocateArray<std::int32_t>(data.bgr_s32_o, pixels * 3);
        allocateArray<float>(data.yuv_f32_i.y, pixels);
        allocateArray<float>(data.yuv_f32_i.u, pixels);
        allocateArray<float>(data.yuv_f32_i.v, pixels);
        allocateArray<float>(data.yuv_f32_o.y, pixels);
        allocateArray<float>(data.yuv_f32_o.u, pixels);
        allocateArray<float>(data.yuv_f32_o.v, pixels);
        allocateArray<std::uint8_t>(data.yuv_u8_i.y, pixels);
        allocateArray<std::uint8_t>(data.yuv_u8_i.u, pixels);
        allocateArray<std::uint8_t>(data.yuv_u8_i.v, pixels);
        allocateArray<std::uint8_t>(data.yuv_u8_o.y, pixels);
        allocateArray<std::uint8_t>(data.yuv_u8_o.u, pixels);
        allocateArray<std::uint8_t>(data.yuv_u8_o.v, pixels);
        allocateArray<std::uint8_t>(data.bgr_u8_o, pixels * 3);
    } catch (...) {
        // A failed constructor will not call ~Frame(). Clean up every
        // successful allocation before allowing bad_alloc to propagate.
        releaseStorage(data);
        throw;
    }
    return 0;
}

Frame::~Frame() {
    releaseStorage(data);
}

int Frame::RawMemToFrame(void* source, int length) {
    if (!validRawLength(info, length) || (length != 0 && source == nullptr))
        return -1;
    if (data.raw_u8_i == nullptr) return -2;
    if (length != 0)
        std::memcpy(data.raw_u8_i, source, static_cast<std::size_t>(length));
    return 0;
}

int Frame::ReadFileToFrame(std::string filename, int size) {
    if (!validRawLength(info, size) || data.raw_u8_i == nullptr) return -1;
    std::FILE* file = std::fopen(filename.c_str(), "rb");
    if (file == nullptr) return -1;
    // Preserve the SDK convention: 1 complete record, 0 short/empty record,
    // and -1 for an invalid request or failure to open the input file.
    const auto records = std::fread(data.raw_u8_i, static_cast<std::size_t>(size), 1, file);
    std::fclose(file);
    return static_cast<int>(records);
}
