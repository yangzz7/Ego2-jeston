#pragma once
#include "common/types.h"
#include <cstdint>
#include <memory>
namespace camera_tuner {
struct CudaDisplayParameters {
    int black, rggb;
    float wb[3], ccm[9];
    BayerNrPrms bayer;
    LumaNrPrms luma;
    ChromaNrPrms chroma;
    FalseColorPrms falseColor;
    float encode[1024];
    float blackPoint, contrast;
};
// Per-eye streams and scratch. No sensor/algorithm buffers are modified.
class CudaDisplay {
public:
    CudaDisplay();
    ~CudaDisplay();
    CudaDisplay(const CudaDisplay&) = delete;
    CudaDisplay& operator=(const CudaDisplay&) = delete;
    void process(const uint8_t* raw, int width, int height,
                 const CudaDisplayParameters& parameters, uint8_t* bgr);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
