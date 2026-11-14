# Cyperstereo 双目 ISP 移植包

本包从 `camera_tuner --serial s200032` 的图像处理链独立整理。默认处理效果与调参工具的“自定义 ISP”一致：黑电平 → Bayer 降噪 → 白平衡 → 去马赛克 → 颜色矩阵/饱和度 → 亮度/色度降噪 → 伪色抑制 → sRGB/色调 → 可选锐化。

包内包含所有用到的 SDK 源码，构建和运行不依赖原来的 CyperstereoSDK 目录。不含 GUI、相机采集、ROS、SLAM、Rerun；不打开设备、不调整硬件曝光/增益。原 camera_tuner 和 Ego2 项目不需要因本包的生成而改动。

## 1. 构建

依赖：CMake 3.16+、C++17 编译器、OpenCV 4 开发库（core、imgproc）、线程库。离线 PNG 示例还需要 imgcodecs。Python 接口需要 Python 3.10+、NumPy；它通过 C ABI 调用本机编译的共享库，不依赖 Python 的 OpenCV/pybind11。

```bash
cd /home/taoqiu/cyperstereo_isp_portable
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

关键产物：

| 文件 | 用途 |
| --- | --- |
| `build/libcyperstereo_isp.a` | C++ 静态库 |
| `build/libcyperstereo_isp_c.so` | C/Python 调用的共享库 |
| `build/isp_convert` | 将保存的两路 RAW8 离线处理为 PNG |
| `configs/s200032.yaml` | 与 tuner 相同的默认配置，可用 tuner 导出的完整 YAML 替换 |
| `include/cyperstereo_isp/processor.h` | C++ 接口 |
| `include/cyperstereo_isp/c_api.h` | C ABI 接口及缓冲区约定 |
| `python/cyperstereo_isp.py` | NumPy 包装器 |

当前验证平台是本机 Linux x86_64、GCC 11、系统 OpenCV 4.5.4。源码包含 x86/ARM 分支，其他架构需用目标工具链重新构建和验证。默认不使用 `-march=native`，避免产物只能在构建机 CPU 运行。若仅在本机部署，可配置 `-DCISP_NATIVE_OPTIMIZATION=ON`；默认构建的吞吐量不等同于 tuner 开启本机优化时的吞吐量。

激活 Conda 时，默认优先系统 OpenCV。使用其他 OpenCV 可显式指定 `-DOpenCV_DIR=/你的/OpenCV/cmake目录`，并保证编译器和该 OpenCV 的 C++ ABI 兼容。移植到另一台机器时复制源码后重新生成 `build`，不要复用包含旧绝对路径的 CMake 缓存。

## 2. 输入、输出必须满足的约定

- 主要输入是 S2 原生 **1280×1024×2、uint8** 传输缓冲区：每个像素依次为 C1 Bayer 字节、C2 Bayer 字节。虽然 UVC 格式标签叫 YUYV，它不是普通 YUV；不得先做 YUYV→RGB 转换。
- 最后一行包含元数据，必须原样保留。接口从中读取硬件版本、固件、设备时间戳及曝光/增益遥测。支持硬件 2、固件 3/4/5/6；异常元数据直接报错，不猜相位。
- C1 是低字节通道，C2 是高字节通道。物理左右镜头身份由原项目标定/遮挡确认，包不交换通道。
- 输出是 `(C1 BGR8, C2 BGR8)`，每眼 **1280×1024×3**，没有去畸变、裁剪、缩小或 JPEG 编码。OpenCV 直接使用 BGR；交给 Rerun/PIL 前按目标 API 需要转换成 RGB 一次。
- ISP 在工作副本中将元数据最后一行替换为同 Bayer 相位的 H−3 行；返回/保存的原始 Bayer 不会被改写。几何算法仍应屏蔽原项目规定的无效底行。
- **不要先去马赛克或白平衡再传入，也不要在输出之后再运行原来的白平衡/Gamma。**
- 每台相机保持一个长期存活的处理器，以保留两眼独立的自动白平衡历史。换相机/重新开始采集可 `reset()`。不要每一帧创建处理器。
- C++/Python 调用均为同步：输入在调用期间不能被采集线程覆盖。返回图像拥有独立存储，下一次调用不会改写上一帧。

自动白平衡有时间历史。同一帧要逐像素复现 tuner 的某个历史输出，应在 tuner 中锁定白平衡后保存配置，或使用该快照 `metadata.json` 中实际的 `actual_white_balance_rgb` 并关闭自动估计。只复制自动模式配置后处理一张孤立帧，可能与已运行数秒的 tuner 略有差异。

## 3. Python / Ego2 接入

将包内 `python` 目录加入 `PYTHONPATH`：

```bash
export PYTHONPATH=/home/taoqiu/cyperstereo_isp_portable/python${PYTHONPATH:+:$PYTHONPATH}
```

在现有相机循环外创建 ISP，在 `cap.read()` 之后处理。以下为集成片段，`cap`、`running` 和下游处理由原项目提供：

```python
from cyperstereo_isp import Isp

# cap 仍需配置为 YUYV 标签、1280×1024、CAP_PROP_CONVERT_RGB=0。
with Isp(config="/home/taoqiu/cyperstereo_isp_portable/configs/s200032.yaml") as isp:
    while running:
        ok, transport = cap.read()
        if not ok:
            raise RuntimeError("Camera read failed")
        c1_bgr, c2_bgr = isp.process_transport(transport)
        # 将两张 BGR 图交给原检测/显示逻辑。
        # 需要 Rerun 时：rgb = c1_bgr[:, :, ::-1].copy()
```

Ego2 的具体接入位置为 `ego2/common/sources.py` 中 Cyper 输入分支：用 `isp.process_transport(frame)` **替换原来的 `decode_cyper(...)` 和随后 `white_balance(...)` 两步**。初始化放在 `Source.open()` 或其初始化阶段，释放放在 `Source.close()`。保留原项目的序列号选择、采集、IMU/元数据解码、时间戳、原始传输保存和 C1/C2 标定绑定。

本次只提供接口与接入说明，没有修改 Ego2。不同项目应先通过同帧 A/B 检查颜色与通道，再验证检测质量和处理速度。处理速度跟不上采集时，可用有界的最新帧队列处理预览；需要完整采集时，应把原始录制与 ISP 预览分开，不能让预览丢帧策略影响原始录制。

Python 的 `load_config(path)` 可加载 tuner 导出的 YAML；`reset()` 清空 AWB 历史；`last_white_balance_rgb` 返回最近一帧两眼的实际 RGB 增益。`library=` 或环境变量 `CYPERSTEREO_ISP_LIBRARY` 可指定共享库路径，默认是包内 `build/libcyperstereo_isp_c.so`。路径解析不依赖工作目录。

## 4. C++ 接入

最直接的方式是将整个目录复制到项目的 `third_party/cyperstereo_isp`：

```cmake
set(CISP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CISP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/cyperstereo_isp)
target_link_libraries(your_app PRIVATE CyperStereo::isp)
```

```cpp
#include <cyperstereo_isp/processor.h>

auto params = cyperstereo_isp::loadParameters("configs/s200032.yaml");
cyperstereo_isp::Processor isp(params);  // 保留到采集结束

// 以下位于循环内；data/byte_count/row_stride 是当前有效传输缓冲区。
auto output = isp.processTransport(data, byte_count, row_stride);
const cv::Mat& c1_bgr = output.bgr[0];
const cv::Mat& c2_bgr = output.bgr[1];
```

运行时调参：复制 `isp.parameters()`，改 `eyes[0/1]` 中的字段，再 `isp.setParameters(params)`。`linked` 是 tuner GUI 的编辑行为标记，库不会根据该标记自动改写另一眼；C++ 调参时需要显式设置两眼。自动白平衡开启时，手动 `whiteBalance` 字段不生效；`gamma` 只在 `ToneMode::Power` 生效。

`processRaw(RawStereoFrame)` 是已有两路 Bayer 平面时的高级接口：平面须完整保留原始坐标和元数据末行，填写真实硬件/固件版本及可用的增益遥测；没有有效遥测时降噪按增益 1 处理。优先使用 `processTransport`，它会直接从帧内解析信息，避免手工写错相位和增益。

也可以安装后通过 CMake 查找：

```bash
cmake --install build --prefix /你的/安装目录
```

```cmake
find_package(CyperStereoIsp 1 CONFIG REQUIRED)
target_link_libraries(your_app PRIVATE CyperStereo::isp)
```

该安装规则安装 C++/C ABI 和默认配置；Python 使用源码包中的 `python` 文件，必要时显式传入安装后的共享库与配置路径。同一程序若已经编译了同名的 SDK ISP、全局 Frame 或 Easylogging++ 实现，应先统一依赖；不要再编译两套相同的全局定义。Python 共享库隐藏了内部 SDK 符号，只对外提供 C ABI。

## 5. 离线验证，无需连接相机

使用 tuner“保存快照”得到的 `c1_bayer.raw8` 和 `c2_bayer.raw8`。两文件必须来自同一帧，并包含完整元数据行。输出目录必须尚不存在：

```bash
./build/isp_convert configs/s200032.yaml \
  /快照目录/c1_bayer.raw8 /快照目录/c2_bayer.raw8 ./cpp-output

python3 examples/python_pipeline.py \
  /快照目录/c1_bayer.raw8 /快照目录/c2_bayer.raw8 \
  --output ./python-output --png
```

C++ 示例输出全尺寸 PNG 和请求参数。Python 示例默认输出无损 `.npy`；加 `--png` 时额外需要 `cv2`，输出 PNG。示例不会打开设备。

## 6. 源码与维护边界

- `src/tunable_isp.cpp`、`src/isp_frame_storage.cpp` 与原 tuner 完全一致；`vendor` 内包含用到的 SDK ISP、格式/元数据头、Fast 对照后端与日志依赖。
- `src/parameters.cpp` 保留 tuner 的严格配置格式及参数范围，文件读写改成 C++17，移除 Qt。
- `src/processor.cpp` 提供持久处理器和内存传输解码；`src/c_api.cpp`、`python` 提供语言桥接。
- 兼容头沿用 `camera_tuner` 命名空间；移除了 GUI 快照写入函数声明，不导出不存在的功能。
- 默认 `mode: tunable`；`minimal`、`sdk_fast` 仅作对照，只有 tunable 应用可调参数。Fast 后端保留上游环境变量/线程策略，可能尝试调整线程调度；需要可调图像效果时保持默认 tunable。
- `SOURCE_MANIFEST.json` 记录复制来源与哈希；`VALIDATION.md` 记录本次构建和同帧对比；许可与来源见 `THIRD_PARTY_NOTICES.md`、`LICENSE`、`LICENSES/`。

参数格式与 tuner 一致，因此可以直接使用面板“保存”导出的配置。这个包不会监视 tuner 的实时滑条，也不会主动更新任何其他项目；调用方通过加载配置或 C++ 参数接口决定使用哪些参数。
