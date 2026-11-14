# Jetson RAW 图像转换 CUDA 后端

2026-11-06，迭代 gpu-capture-20261106-01。

四个实时模式（手势、AprilGrid、VIO、三维重建）默认在 Jetson 使用同一个 CUDA 双目转换实现。USB/V4L2 读取、帧缓冲复制、元数据/IMU 解析仍在 CPU；左右目图像字节拆分、元数据末行同奇偶替换、Bayer→BGR 双线性插值在 GPU。图像白平衡、校正和后续算法保持原实现。回放/离线录制解码保留 CPU 参考路径。

Python Source 在打开设备时建立解码器。C++ 后端让 SDK 仅解析元数据行，并持有独立的完整 transport 副本；处理线程在 GPU 上从 packed RAW 直接生成 cam0/C1 与 cam1/C2。显示 RAW 环直接写原 packed 字节，避免 CPU 拆分后再次交织。原始 transport 不被修改。

采用持久 CUDA mapped pinned 缓冲区；同步完成后复制到调用者独立持有的 BGR 缓冲区，避免下一帧覆盖算法/录制正在使用的图像。CUDA 阻塞事件避免主动忙等。每个相机进程各自持有上下文。

转换只使用整数加法、移位和原 Bayer 样本；必须与现有 OpenCV 双线性输出逐像素一致，不使用显示增强 ISP 的近似输出。参考行为来自 OpenCV 原始实现：https://github.com/opencv/opencv/blob/4.12.0/modules/imgproc/src/demosaicing.cpp 。独立实现已对 Jetson 的 Python OpenCV 4.12 与 C++ OpenCV 4.8 验证。

## 构建与检查

在 Jetson 的 Ego2-jetson 根目录：

```bash
.venv-jetson/bin/python -m ego2.common.build_capture_cuda
EGO2_TEST_CAPTURE_CUDA=1 .venv-jetson/bin/python -m unittest tests.test_capture_cuda
```

原生两个后端按原有构建入口重建；其 CMake 增加 dl 库依赖和 test_capture_cuda 验证程序。原生测试在 build/reconstruction/bin/test_capture_cuda，默认库在 build/cyper_capture/libego2_cyper_cuda.so。

在 Jetson 上通过 /etc/nv_tegra_release 检测默认启用 CUDA。若显式设置 EGO2_CAPTURE_BACKEND=cpu，则全部实时模式返回原 CPU 路径；EGO2_CAPTURE_BACKEND=cuda 强制 CUDA，失败时给出错误，不静默回退。EGO2_CAPTURE_CUDA_LIBRARY 可以覆盖库路径用于隔离验证。以上环境变量须设置在 Jetson 的应用进程环境中，而非仅设置在电脑 SSH 客户端。

启动日志标记：`[capture-convert] cuda: RAW lanes + Bayer to BGR; USB/metadata on CPU`。

## 验证范围与限制

测试覆盖固件 03–07、RG/BG/GR/GB 四种 OpenCV Bayer 名称、奇偶尺寸、图像边界、极端像素、输入只读、独立输出生命周期、四路并发、坏帧拒绝，以及 Python/C++ 元数据与原始显示环一致性。相关采集、显示和录制回归共 51 项。真实 S200097 固件 2/7 的 60 对帧与原 CPU 加白平衡输出逐像素相同。

最终转换微基准包含分配和内存传输：预热后单路 CPU 用时约 5.3→2.2 ms/对，总耗时约 5.3→4.1 ms/对；四路并发每对 CPU 用时约 5.3→2.3 ms，但每批四对总耗时约 5.8→9.9 ms。它减轻 CPU 负担，并不保证每种负载下延迟更低。未调整电源模式/固定时钟，数字随动态频率变化；该微基准不是整机 FPS。

只有一台实体相机参与此次检查。四台相机同时运行时的 USB、GPU 竞争、显示/算法负载仍需现场验收。手势推理的 DISPLAY/驱动回退问题属于独立工作，本次未改动推理代码。
