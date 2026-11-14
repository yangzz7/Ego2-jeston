#include "tunable_isp.h"

#include "cyperstereo_api.h"
#include "common/pipeline.h"
#ifdef CISP_CUDA_DISPLAY
#include "cuda_display.h"
#endif

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace camera_tuner {
namespace {

using Bayer = cyperstereo::BayerConversion;
constexpr int kWorkingMaximum = 1023;

// The legacy module registry is process-global and not synchronized. Keep our
// construction/reconstruction sequential even if several tuners are created.
std::mutex& registryMutex() {
    static std::mutex mutex;
    return mutex;
}

CfaTypes physicalCfa(Bayer bayer) {
    return bayer == Bayer::kColorBayerBg2Bgr ? CfaTypes::RGGB : CfaTypes::BGGR;
}

double sensorGain(const Telemetry& telemetry) {
    return telemetry.valid && std::isfinite(telemetry.gain) && telemetry.gain >= 1.0
        ? std::clamp(telemetry.gain, 1.0, 8.0) : 1.0;
}

struct EyeContext {
    IspPrms parameters{};
    std::unique_ptr<::Frame> frame;
    std::unique_ptr<IspPipeline> pipeline;
    std::array<double, 3> whiteBalance{{1.0, 1.0, 1.0}};
    bool whiteBalanceInitialized = false;

    void prepare(int width, int height, CfaTypes cfa, bool cpu = true) {
        if ((frame || !cpu) && parameters.info.width == width && parameters.info.height == height &&
            parameters.info.cfa == cfa)
            return;
        // Never call MakePipe twice on an existing SDK pipeline: it appends.
        std::lock_guard<std::mutex> lock(registryMutex());
        parameters = IspPrms{};
        parameters.info.width = width;
        parameters.info.height = height;
        parameters.info.bpp = 16;
        parameters.info.max_val = kWorkingMaximum;
        parameters.info.mipi_packed = false;
        parameters.info.cfa = cfa;
        parameters.info.dt = RawDataTypes::RAW8;
        parameters.info.domain = ColorDomains::RAW;
        if (cpu) {
            frame = std::make_unique<::Frame>(parameters.info);
            pipeline = std::make_unique<IspPipeline>();
            if (pipeline->MakePipe({"blc", "bayernr", "wbgain", "demoasic", "ccm",
                                    "lumanr", "chromanr", "falsecolor"}) != 0)
                throw std::runtime_error("Cannot construct the tunable ISP pipeline");
        }
        whiteBalance = {{1.0, 1.0, 1.0}};
        whiteBalanceInitialized = false;
    }
};

double median(std::vector<double>& values) {
    const auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
}

void updateWhiteBalance(EyeContext& context, const cv::Mat& raw,
                        const EyeParameters& parameters, Bayer bayer) {
    if (!parameters.autoWhiteBalance) {
        context.whiteBalance = parameters.whiteBalance;
        context.whiteBalanceInitialized = true;
        return;
    }
    std::vector<double> redRatios, blueRatios;
    const bool rggb = physicalCfa(bayer) == CfaTypes::RGGB;
    // Stay out of the metadata row and away from nearly black/clipped CFA
    // samples. Ratios are calculated after the selected RAW black pedestal.
    for (int y = 0; y + 1 < raw.rows - 1; y += 8) {
        const auto* row0 = raw.ptr<unsigned char>(y);
        const auto* row1 = raw.ptr<unsigned char>(y + 1);
        for (int x = 0; x + 1 < raw.cols; x += 8) {
            const double r0 = rggb ? row0[x] : row1[x + 1];
            const double b0 = rggb ? row1[x + 1] : row0[x];
            const double g00 = row0[x + 1], g10 = row1[x];
            if (std::max({r0, b0, g00, g10}) >= 250.0) continue;
            const double r = r0 - parameters.blackLevel;
            const double b = b0 - parameters.blackLevel;
            const double g = (g00 + g10) * 0.5 - parameters.blackLevel;
            if (std::min({r, g, b}) < 4.0 || (r + 2 * g + b) * 0.25 < 24.0)
                continue;
            redRatios.push_back(g / r);
            blueRatios.push_back(g / b);
        }
    }
    if (redRatios.size() < 4) return;
    const std::array<double, 3> estimate{{
        std::clamp(median(redRatios), 0.6, 2.6), 1.0,
        std::clamp(median(blueRatios), 0.6, 2.6)}};
    if (!context.whiteBalanceInitialized) {
        context.whiteBalance = estimate;
    } else {
        for (int channel = 0; channel < 3; ++channel)
            context.whiteBalance[channel] +=
                0.05 * (estimate[channel] - context.whiteBalance[channel]);
    }
    context.whiteBalanceInitialized = true;
}

void configure(EyeContext& context, const EyeParameters& requested, double gain) {
    auto& p = context.parameters;
    p.blc = static_cast<int>(std::lround(requested.blackLevel * 4.0));
    p.wb_gains.d65_gain[0] = static_cast<float>(context.whiteBalance[0]);
    p.wb_gains.d65_gain[1] = static_cast<float>(context.whiteBalance[1]);
    p.wb_gains.d65_gain[2] = static_cast<float>(context.whiteBalance[1]);
    p.wb_gains.d65_gain[3] = static_cast<float>(context.whiteBalance[2]);

    // Saturation preserves linear Rec.601 luminance: S = sI + (1-s) 1 w^T.
    // Compose S*C, both R-major, rather than rotate U/V or swap R/B twice.
    constexpr double weights[3] = {0.299, 0.587, 0.114};
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            double combined = 0.0;
            for (int k = 0; k < 3; ++k) {
                const double saturation = (row == k ? requested.saturation : 0.0) +
                    (1.0 - requested.saturation) * weights[k];
                const double ccm = requested.ccmEnabled
                    ? requested.ccm[3 * k + col] : (k == col ? 1.0 : 0.0);
                combined += saturation * ccm;
            }
            p.ccm_prms.ccm[row][col] = static_cast<float>(combined);
        }
    }

    const double gainFraction = (gain - 1.0) / 7.0;
    p.bayer_nr_prms.threshold_base = static_cast<int>(16 + 8 * gainFraction);
    p.bayer_nr_prms.threshold_signal_q8 = static_cast<int>(4 + 2 * gainFraction);
    p.bayer_nr_prms.center_weight = gain >= 4.0 ? 3 : 4;
    p.bayer_nr_prms.strength_q8 = static_cast<int>(std::lround(requested.bayerDenoise * 256));
    p.luma_nr_prms.eps = static_cast<int>(128 + 320 * gainFraction);
    p.luma_nr_prms.strength_q8 = static_cast<int>(std::lround(requested.lumaDenoise * 256));
    p.chroma_nr_prms.neutral_strength_q8 =
        static_cast<int>(std::lround(requested.chromaDenoise * 256));
    p.chroma_nr_prms.saturated_strength_q8 =
        static_cast<int>(std::lround(requested.chromaDenoise * 80));
    p.chroma_nr_prms.neutral_outlier_strength_q8 =
        static_cast<int>(std::lround(requested.chromaDenoise * 256));
    p.false_color_prms.max_attenuation_q8 =
        static_cast<int>(std::lround(requested.falseColor * 256));
}

void encodeOutput(const EyeContext& context, const EyeParameters& p, cv::Mat& bgr) {
    // This is the only display transfer. The upstream pipeline contains no
    // rgbgamma/YUV/black-point stages. RAW8 <<2 retains arithmetic precision;
    // it does not recover the sensor bits discarded by the FPGA.
    std::array<float, kWorkingMaximum + 1> encode{};
    const double fullScale = 1020.0 - context.parameters.blc;
    const double ev = std::exp2(p.exposureEv);
    for (int value = 0; value <= kWorkingMaximum; ++value) {
        const double linear = std::clamp(value * ev / fullScale, 0.0, 1.0);
        double output = linear;
        if (p.tone == ToneMode::SRgb)
            output = linear <= 0.0031308 ? 12.92 * linear
                : 1.055 * std::pow(linear, 1.0 / 2.4) - 0.055;
        else if (p.tone == ToneMode::Power)
            output = std::pow(linear, 1.0 / p.gamma);
        encode[value] = static_cast<float>(output);
    }
    bgr.create(context.frame->info.height, context.frame->info.width, CV_8UC3);
    const auto* input = reinterpret_cast<const int32_t*>(context.frame->data.bgr_s32_i);
    const double blackPoint = p.blackPoint / 255.0;
    for (int y = 0; y < bgr.rows; ++y) {
        auto* output = bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < bgr.cols; ++x) {
            const auto* pixel = input + 3 * (static_cast<size_t>(y) * bgr.cols + x);
            double colors[3] = {
                encode[std::clamp(pixel[0], 0, kWorkingMaximum)],
                encode[std::clamp(pixel[1], 0, kWorkingMaximum)],
                encode[std::clamp(pixel[2], 0, kWorkingMaximum)]};
            const double luminance = 0.114 * colors[0] + 0.587 * colors[1] + 0.299 * colors[2];
            const double t = std::clamp((luminance - blackPoint) / (1.0 - blackPoint), 0.0, 1.0);
            const double target = (1.0 - p.contrast) * t + p.contrast * t * t * (3.0 - 2.0 * t);
            double ratio = luminance > 0.0 ? target / luminance : 0.0;
            const double maximum = std::max({colors[0], colors[1], colors[2]});
            if (maximum > 0.0) ratio = std::min(ratio, 1.0 / maximum);
            for (int channel = 0; channel < 3; ++channel)
                output[x][channel] = cv::saturate_cast<unsigned char>(colors[channel] * ratio * 255.0);
        }
    }
}

void sharpenLuminance(cv::Mat& bgr, double amount) {
    if (amount <= 0.0) return;
    cv::Mat gray, floating, blurred;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(floating, CV_32F);
    cv::GaussianBlur(floating, blurred, cv::Size(3, 3), 0.8);
    for (int y = 0; y < bgr.rows; ++y) {
        auto* row = bgr.ptr<cv::Vec3b>(y);
        const auto* original = floating.ptr<float>(y);
        const auto* smooth = blurred.ptr<float>(y);
        for (int x = 0; x < bgr.cols; ++x) {
            const auto pixel = row[x];
            // Bounded, common luma delta preserves R-G and B-G; do not
            // introduce channel-specific clipping or aggressive halos.
            double delta = std::clamp(static_cast<double>(original[x] - smooth[x]), -12.0, 12.0) * amount;
            delta = std::clamp(delta, -static_cast<double>(std::min({pixel[0], pixel[1], pixel[2]})),
                                255.0 - std::max({pixel[0], pixel[1], pixel[2]}));
            const int rounded = static_cast<int>(std::lround(delta));
            for (int channel = 0; channel < 3; ++channel)
                row[x][channel] = cv::saturate_cast<unsigned char>(pixel[channel] + rounded);
        }
    }
}

void calculateMetrics(const cv::Mat& bgr, EyeMetrics& metrics) {
    // Last row was reconstructed to keep Bayer geometry; exclude it from
    // measurements. Clipping counts output pixels with any channel at 255.
    const cv::Mat valid = bgr.rowRange(0, bgr.rows - 1);
    cv::Mat gray;
    cv::cvtColor(valid, gray, cv::COLOR_BGR2GRAY);
    std::size_t clipped = 0;
    for (int y = 0; y < valid.rows; ++y) {
        const auto* pixels = valid.ptr<cv::Vec3b>(y);
        const auto* luminance = gray.ptr<unsigned char>(y);
        for (int x = 0; x < valid.cols; ++x) {
            ++metrics.histogram[luminance[x]];
            clipped += std::max({pixels[x][0], pixels[x][1], pixels[x][2]}) == 255;
        }
    }
    metrics.clippedPercent = 100.0 * clipped / valid.total();
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar mean, deviation;
    cv::meanStdDev(laplacian, mean, deviation);
    metrics.sharpness = deviation[0] * deviation[0];
}

}  // namespace

struct TunableIsp::Impl {
    std::mutex mutex;
    std::array<EyeContext, 2> eyes;
    bool displayOnly = false, useCuda = false;
#ifdef CISP_CUDA_DISPLAY
    std::array<std::unique_ptr<CudaDisplay>, 2> gpu;
#endif
    std::unique_ptr<cyperstereo::IspProcessor> fast;
    int fastFirmware = -1;
    cv::Size fastSize;
    std::mutex jobMutex;
    std::condition_variable ready, completed;
    std::function<void()> job;
    std::exception_ptr jobError;
    bool stopping = false;
    bool busy = false;
    std::thread rightWorker;

    Impl() {
        // The SDK quality module would otherwise log every stage of every
        // frame. Keep errors/warnings; construction remains serialized.
        static std::once_flag logging;
        std::call_once(logging, [] {
            el::Loggers::reconfigureAllLoggers(el::Level::Info,
                el::ConfigurationType::Enabled, "false");
            el::Loggers::reconfigureAllLoggers(el::Level::Debug,
                el::ConfigurationType::Enabled, "false");
        });
        rightWorker = std::thread([this] {
            std::unique_lock<std::mutex> lock(jobMutex);
            for (;;) {
                ready.wait(lock, [this] { return busy || stopping; });
                if (stopping) return;
                auto work = std::move(job);
                lock.unlock();
                std::exception_ptr error;
                try { work(); } catch (...) { error = std::current_exception(); }
                lock.lock();
                jobError = error;
                busy = false;
                completed.notify_one();
            }
        });
    }

    ~Impl() {
        {
            std::lock_guard<std::mutex> lock(jobMutex);
            stopping = true;
        }
        ready.notify_one();
        if (rightWorker.joinable()) rightWorker.join();
    }

    void submitRight(std::function<void()> work) {
        std::lock_guard<std::mutex> lock(jobMutex);
        jobError = nullptr;
        job = std::move(work);
        busy = true;
        ready.notify_one();
    }

    void waitRight() {
        std::unique_lock<std::mutex> lock(jobMutex);
        completed.wait(lock, [this] { return !busy; });
        if (jobError) std::rethrow_exception(jobError);
    }
};

TunableIsp::TunableIsp() : impl_(std::make_unique<Impl>()) {}
TunableIsp::~TunableIsp() = default;

void TunableIsp::setDisplayBackend(bool cuda) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
#ifdef CISP_CUDA_DISPLAY
    if (cuda) for (auto& eye : impl_->gpu) eye = std::make_unique<CudaDisplay>();
#else
    if (cuda) throw std::runtime_error("This ISP library was built without CUDA display support");
#endif
    impl_->displayOnly = true;
    impl_->useCuda = cuda;
}

void TunableIsp::reset() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (auto& eye : impl_->eyes) {
        eye.whiteBalance = {{1.0, 1.0, 1.0}};
        eye.whiteBalanceInitialized = false;
    }
    impl_->fast.reset();
    impl_->fastFirmware = -1;
}

ProcessedFrame TunableIsp::process(std::shared_ptr<const RawStereoFrame> source,
                                  const Parameters& requestedParameters) {
    const Parameters parameters = requestedParameters;
    validateParameters(parameters);
    if (!source || source->hardwareVersion != 2 ||
        source->firmwareVersion < 3 || source->firmwareVersion > 7)
        throw std::invalid_argument("Expected SmartSens hardware 2, firmware 3..7");
    for (const auto& raw : source->raw) {
        if (raw.type() != CV_8UC1 || raw.cols < 8 || raw.rows < 8 ||
            (raw.cols & 1) || (raw.rows & 1) || raw.size() != source->raw[0].size())
            throw std::invalid_argument("Expected equal even-sized RAW8 stereo planes, at least 8x8");
    }
    if (parameters.mode == PipelineMode::SdkFast &&
        ((source->raw[0].cols & 3) || (source->raw[0].rows & 3)))
        throw std::invalid_argument("SDK fast comparison requires dimensions divisible by four");

    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto started = std::chrono::steady_clock::now();
    ProcessedFrame result;
    result.source = std::move(source);
    result.parameters = parameters;  // one revision for both eyes
    std::array<cv::Mat, 2> working;
    std::array<Bayer, 2> bayers;
    for (int eye = 0; eye < 2; ++eye) {
        working[eye] = result.source->raw[eye].clone();
        // Replace the transport metadata row only in this processing copy.
        // y-2 has the SAME Bayer phase; copying y-1 would swap R/G/B phases.
        working[eye].row(working[eye].rows - 3).copyTo(working[eye].row(working[eye].rows - 1));
        bayers[eye] = cyperstereo::SelectBayerConversion(
            result.source->hardwareVersion, result.source->firmwareVersion, eye);
    }
    if (parameters.mode == PipelineMode::SdkFast) {
        if (!impl_->fast || impl_->fastFirmware != result.source->firmwareVersion ||
            impl_->fastSize != working[0].size()) {
            impl_->fast = std::make_unique<cyperstereo::IspProcessor>(cyperstereo::IspMode::kFastBalancedBgr888);
            impl_->fastFirmware = result.source->firmwareVersion;
            impl_->fastSize = working[0].size();
        }
        impl_->fast->ApplyParallel({
            {working[0], result.bgr[0], "tuner-cam1", sensorGain(result.source->telemetry[0]), bayers[0]},
            {working[1], result.bgr[1], "tuner-cam2", sensorGain(result.source->telemetry[1]), bayers[1]},
        });
        for (auto& metrics : result.metrics)
            metrics.whiteBalance.fill(std::numeric_limits<double>::quiet_NaN());
    } else if (parameters.mode == PipelineMode::Minimal) {
        for (int eye = 0; eye < 2; ++eye) {
            cv::cvtColor(working[eye], result.bgr[eye],
                bayers[eye] == Bayer::kColorBayerBg2Bgr ? cv::COLOR_BayerBG2BGR : cv::COLOR_BayerRG2BGR);
        }
    } else {
        // Register/construct both contexts sequentially BEFORE dispatch.
        // Every camera then retains its own Frame/AWB history and thread-local
        // module scratch, without spawning threads for each acquired frame.
        for (int eye = 0; eye < 2; ++eye)
            impl_->eyes[eye].prepare(working[eye].cols, working[eye].rows, physicalCfa(bayers[eye]), !impl_->useCuda);
        const auto processEye = [&](int eye) {
            auto& context = impl_->eyes[eye];
            const auto& requested = parameters.eyes[eye];
            updateWhiteBalance(context, working[eye], requested, bayers[eye]);
            configure(context, requested, sensorGain(result.source->telemetry[eye]));
#ifdef CISP_CUDA_DISPLAY
            if (impl_->useCuda) {
                CudaDisplayParameters p{};
                p.black = context.parameters.blc;
                p.rggb = physicalCfa(bayers[eye]) == CfaTypes::RGGB;
                for (int c = 0; c < 3; ++c) p.wb[c] = static_cast<float>(context.whiteBalance[c]);
                for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c)
                    p.ccm[3*r+c] = context.parameters.ccm_prms.ccm[r][c];
                p.bayer = context.parameters.bayer_nr_prms;
                p.luma = context.parameters.luma_nr_prms;
                p.chroma = context.parameters.chroma_nr_prms;
                p.falseColor = context.parameters.false_color_prms;
                p.blackPoint = requested.blackPoint / 255.0;
                p.contrast = requested.contrast;
                for (int v = 0; v <= kWorkingMaximum; ++v) {
                    double linear = std::clamp(v * std::exp2(requested.exposureEv) / (1020.0-p.black), 0.0, 1.0);
                    double encoded = linear;
                    if (requested.tone == ToneMode::SRgb)
                        encoded = linear <= 0.0031308 ? 12.92*linear : 1.055*std::pow(linear,1.0/2.4)-0.055;
                    else if (requested.tone == ToneMode::Power) encoded = std::pow(linear,1.0/requested.gamma);
                    p.encode[v] = static_cast<float>(encoded);
                }
                result.bgr[eye].create(working[eye].size(), CV_8UC3);
                impl_->gpu[eye]->process(working[eye].ptr<uint8_t>(), working[eye].cols, working[eye].rows,
                                        p, result.bgr[eye].ptr<uint8_t>());
            } else
#endif
            {
            auto* destination = reinterpret_cast<int32_t*>(context.frame->data.raw_s32_i);
            for (int y = 0; y < working[eye].rows; ++y) {
                const auto* row = working[eye].ptr<unsigned char>(y);
                for (int x = 0; x < working[eye].cols; ++x)
                    destination[static_cast<size_t>(y) * working[eye].cols + x] = row[x] << 2;
            }
            if (context.pipeline->RunPipe(context.frame.get(), &context.parameters) != 0)
                throw std::runtime_error("Tunable ISP failed while processing an eye");
            encodeOutput(context, requested, result.bgr[eye]);
            }
            sharpenLuminance(result.bgr[eye], requested.sharpening);
            result.metrics[eye].whiteBalance = context.whiteBalance;
            if (!impl_->displayOnly) calculateMetrics(result.bgr[eye], result.metrics[eye]);
        };
        impl_->submitRight([&] { processEye(1); });
        std::exception_ptr leftError;
        try { processEye(0); } catch (...) { leftError = std::current_exception(); }
        // Join the outstanding job even if C1 failed: its captures refer to
        // this frame's stack. Never return partial or differently tuned eyes.
        impl_->waitRight();
        if (leftError) std::rethrow_exception(leftError);
    }
    if (!impl_->displayOnly && parameters.mode != PipelineMode::Tunable)
        for (int eye = 0; eye < 2; ++eye)
            calculateMetrics(result.bgr[eye], result.metrics[eye]);
    result.processingMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    return result;
}

}  // namespace camera_tuner
