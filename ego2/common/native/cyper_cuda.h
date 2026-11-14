#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
// OpenCV bilinear BGR code names, NOT the physical sensor CFA names.
// RG=0, BG=1, GR=2, GB=3. Outputs are independent C1/cam0 and C2/cam1.
void* ego2_cyper_create(int width, int height, char* error, size_t error_size);
void ego2_cyper_destroy(void* decoder);
int ego2_cyper_decode(void* decoder, const unsigned char* packed, size_t input_step,
                     int phase0, int phase1, unsigned char* cam0, size_t step0,
                     unsigned char* cam1, size_t step1, char* error, size_t error_size);
// Same decoding plus the existing per-eye gray-world white balance (1280x1024).
int ego2_cyper_decode_balanced(void* decoder, const unsigned char* packed, size_t input_step,
                     int phase0, int phase1, unsigned char* cam0, size_t step0,
                     unsigned char* cam1, size_t step1, char* error, size_t error_size);
#ifdef __cplusplus
}
#endif
