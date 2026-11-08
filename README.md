# Ego2-jeston · Jetson Orin Nano 版本

此仓库保存 **2026-11-08 Jetson `/home/yrp/Ego2-jetson` 的实际运行源码**，包含最近的手势识别、IMU 姿态显示及稳定性修复。代码直接取自 Jetson，不是电脑主目录的 Ego2 副本。

## 功能

- **纯 RGB 画面**：单/多相机实时 RGB 预览。
- **单/多相机手势识别**：双目画面与手部骨架；每个部位配有 Mahony 四元数、三轴加速度、三轴陀螺仪及姿态模型。左右手关节由可用相机观测驱动，模型旋转使用对应手部相机的姿态。
- **单/多相机轨迹**：双目惯性 VIO、AprilGrid 初始对齐与轨迹显示。
- **AprilGrid 标定板跟踪与定位**、**三维重建**、腕带标记位姿及录制回放。
- GPU 图像处理与手势推理、独立 IMU 融合、Rerun 可视化和 LeRobot 数据保存。

最新手势修复统一了二维跟踪与三维模型的身份证据；已确认手在短暂弱标签波动时可保持身份，短暂漏检后通过严格匹配可延续编号。没有观测时不伪造手部数据，手形与姿态仍按同一采集时间配对。

## 运行环境

当前设备：Jetson Orin Nano Super Developer Kit，Ubuntu 22.04、L4T 36.5.0、aarch64、Python 3.10.12。完整 Python 包版本见 [环境快照](docs/jetson-environment.json)。其中当前环境使用 MediaPipe 1.2.0、Rerun SDK 0.34.1。

**这是源码仓库，不是可直接复制使用的虚拟环境或刷机镜像。** `.venv-jetson`、`.venv-export`、CUDA/JetPack 系统组件、原生编译产物和设备驱动不包含在 Git 中。新设备需要相匹配的 Jetson 依赖、支持该环境的 MediaPipe 包，以及 Pangolin/ORB-SLAM3/CUDA 扩展的构建。`requirements*.txt` 是原项目依赖要求，不替代 Jetson 平台的完整部署过程。

## 获取源码与模型资源

```bash
git clone https://github.com/yangzz7/Ego2-jeston.git
cd Ego2-jeston
python3 tools/fetch_assets.py
```

下载脚本从上游获取手势模型及 ORB 词袋，并校验 SHA-256 与当前 Jetson 中的文件一致。也可只下载手势模型：

```bash
python3 tools/fetch_assets.py --asset gesture
```

## 在已部署环境中启动

在已有 `.venv-jetson` 和原生依赖的 Jetson 项目目录中：

```bash
bash ./start-sensia.sh
# 指定模式的初始选择
bash ./start-sensia.sh --mode gesture
# 安装环境检查
bash ./check-jetson.sh
```

当前配套电脑通过 `bash ~/jetson-sensia.sh` 连接 Jetson 并打开电脑上的 Rerun。该电脑端连接脚本位于电脑用户目录，不属于此次 Jetson 项目快照。

使用手势姿态显示时，先让相机静止完成 IMU 初始化，再移动相机和手掌。相机标定文件按 SN 放在 `cameras/`；更换设备应使用对应标定参数。

## 目录

| 路径 | 内容 |
| --- | --- |
| `ego2/` | Python 与原生 C++/CUDA 功能源码 |
| `configs/`、`cameras/` | 运行参数、部位姿态约定和相机标定 |
| `calibration_targets/` | 标定板和腕带参数 |
| `third_party/orb_slam3/` | ORB-SLAM3 及 DBoW2、g2o、Sophus、USB 适配源码 |
| `.native/src/Pangolin-0.6/` | 当前设备使用的 Pangolin 源码 |
| `cyperstereo_isp_portable/` | 显示增强模块 |
| `.deployment/scripts/` | 已部署环境的检查脚本 |
| `tests/` | 回归测试和固定样本 |
| `tools/fetch_assets.py` | 模型和词袋下载、校验 |
| `docs/` | 环境、源码校验清单及原功能说明 |

保留的[原功能说明](docs/README-original.md)来自较早的通用版本，其中电脑绝对路径与部分界面描述已过时；当前 Jetson 入口与手势布局以本页及实际代码为准。

## 验证与快照范围

最近一次手势修复在 Jetson 上通过 152 项相关回归检查，正式安装后又通过 26 项关键检查。此次上传保持应用源码字节不变，只整理仓库说明、忽略规则和资源获取方式；不声称已在另一台全新设备完成部署验收。

[源码校验清单](docs/jetson-source-manifest.json)记录设备快照的 SHA-256。未上传虚拟环境、录制视频/数据集、日志、构建缓存、历史修复备份及本地相机选择记录。

## 许可

本项目许可见 [LICENSE](LICENSE)。第三方组件保留各自源码与许可证，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
