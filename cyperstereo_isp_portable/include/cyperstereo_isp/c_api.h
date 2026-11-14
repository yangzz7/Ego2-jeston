#pragma once
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
# if defined(CISP_BUILDING_LIBRARY)
#  define CISP_API __declspec(dllexport)
# else
#  define CISP_API __declspec(dllimport)
# endif
#elif defined(__GNUC__)
# define CISP_API __attribute__((visibility("default")))
#else
# define CISP_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ABI v1. Handles must originate here; never use a handle after destroy.
// Serialize destroy with all other calls. No C++ exceptions cross this API.
// Error text is thread-local, valid until the next API operation on that thread.
CISP_API int cisp_api_version(void);
CISP_API const char* cisp_last_error(void);
CISP_API void* cisp_create(const char* config_path); /* NULL config => defaults */
// Optional ABI-v1 extension: 0=CPU display, 1=CUDA display. Skips unused GUI
// histogram/sharpness metrics; pixel processing and WB remain unchanged.
CISP_API void* cisp_create_display(const char* config_path, int cuda);
CISP_API void cisp_destroy(void* handle);
CISP_API int cisp_load_config(void* handle, const char* config_path);
CISP_API int cisp_reset(void* handle);

// Returns 0 on success, -1 on error. Native input: 1024 rows of 1280 * 2
// interleaved RAW8 bytes WITH metadata row, not conventional YUV. Each output
// is 1024 rows of 1280 * 3 BGR8 bytes. Byte counts must include row padding.
// Buffers must be distinct, writable outputs must not overlap each other/input.
// wb_rgb_six may be NULL, otherwise space for C1 [R,G,B], C2 [R,G,B]; SDK Fast
// reports NaN because it does not expose its internal WB. Caller owns buffers.
CISP_API int cisp_process_transport(
    void* handle, const uint8_t* input, size_t input_bytes, size_t input_stride,
    uint8_t* c1, size_t c1_bytes, size_t c1_stride,
    uint8_t* c2, size_t c2_bytes, size_t c2_stride, double* wb_rgb_six);

#ifdef __cplusplus
}
#endif
