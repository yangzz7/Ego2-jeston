# Ego2：双目鱼眼采集、标定板跟踪与三维重建

本项目为 Cyperstereo 双目鱼眼相机提供标定板跟踪、手势识别与数据采集、腕带位姿估计、三维重建及多相机视觉惯性定位。除腕带位姿外，其他四种模式均可通过统一准备面板启动；原有独立脚本保留。

**最近一次功能更新（2026-10-31）：** `/home/taoqiu/Ego2/run_multi_vio.sh` 打开 **Sensia** 启动面板，提供“实时采集”和“数据回放”两个页签。实时采集页右上角可以选择四种模式，自动发现设备、按 SN 加载标定参数，并记住模式与部位分配；数据回放页可浏览已保存的数据并直接启动 Rerun，无需连接相机。手势与 AprilGrid 支持 1～4 台相机，三维重建仅支持一台并沿用原来的布局。实时采集检查到具有有效双目/IMU 标定文件的新 SN 时，会自动生成缺少的 `camera.yaml`，已有标定文件不会被覆盖。

## 统一启动入口

```bash
# 打开 Sensia：默认启用手动录制，先预览；也可切换到“数据回放”
/home/taoqiu/Ego2/run_multi_vio.sh

# 仅预览，关闭录制功能
/home/taoqiu/Ego2/run_multi_vio.sh --no-record

# 指定面板初始模式，也可在面板里再改
/home/taoqiu/Ego2/run_multi_vio.sh --mode aprilgrid

# 跳过面板，使用记住的模式与分配
/home/taoqiu/Ego2/run_multi_vio.sh --no-setup
```

| 面板模式 / `--mode` | 相机选择规则 | Rerun 布局 |
| --- | --- | --- |
| 单/多相机轨迹 / `vio` | 单台任意部位；2～4 台须包含胸部，胸部作为参考 | 原有左侧 3D 轨迹和初始差值区、右侧双目画面；单台隐藏差值区 |
| 手势识别 / `gesture` | 1～4 台，任意部位组合，无需胸部 | 每台一行，左目与右目两列，叠加手部骨架、手势和状态；底部统一时间轴 |
| AprilGrid 标定板跟踪与定位 / `aprilgrid` | 1～4 台，任意部位组合，无需胸部 | 每台一行，双目画面叠加编号、坐标轴和相对板子的位姿；底部统一时间轴 |
| 三维重建 / `reconstruction` | **仅一台**，任意部位 | **沿用独立重建程序**：左侧 3D 点云，右侧上下两个 RGB 画面，底部时间轴 |

一次运行只选择一种模式，同一个 SN 不可分配给多个部位。VIO 模式需要运动初始化及同一块固定标定板的初始对齐；手势与 AprilGrid 无需 VIO 初始化；三维重建需要自身 VIO 初始化，不等待多相机标定板对齐。

配置 `/home/taoqiu/Ego2/configs/multi_vio.yaml` 中的 `gesture_config`、`aprilgrid_config`、`reconstruction_config` 分别引用现有算法配置。记住的模式与分配保存在同目录的 `multi_vio_selection.json`；旧版本分配文件仍可读取。`--no-setup --mode ...` 可以覆盖记住的模式，但相机组合必须满足新模式规则，不会自动丢弃已选相机。

统一入口**默认启用手动录制，无需添加 `--record`**（保留该参数兼容旧命令）。启动后先预览，不会自动开始保存；`--no-record` 可关闭录制功能、仅保留预览。**点击 Rerun 窗口使其处于前台，按回车**即可打开“录制设置”面板，无需切回终端；主键盘和数字键盘的 Enter 均可。设置存储根目录、数据集文件夹名称后，点击“开始录制”：各相机写入器准备成功后面板自动关闭并开始保存。点击“取消”、关闭面板或按 Esc 仅返回预览，不创建数据集目录。面板内按回车不会直接开始录制。

窗口快捷键适用于当前 Linux X11 桌面上的原生 Rerun，仅匹配本次程序启动的 Viewer；其他软件中的回车不触发，长按不重复弹窗。已有面板时再次触发会将它带到前台。`--connect` 连接已有原生 Viewer 时可从该桌面的 Rerun 窗口触发。Web Viewer、Wayland 或窗口快捷键不可用时，仍可在启动终端按回车打开面板。

两个输入框都可留空：根目录默认 `/home/taoqiu/Ego2/output`，名称默认实际创建时的本地时间（含微秒）。最终路径为 `存储根目录/功能目录/数据集名称`，面板显示完整路径预览。例如根目录留空、手势模式名称填写 `抓取测试01`，保存到 `/home/taoqiu/Ego2/output/gesture/抓取测试01/`。根目录也可用“浏览”选择；不存在的父目录在开始录制时创建。已有文件夹不覆盖，路径不可写或空间不足时在面板内提示。

预览和面板操作不会重启相机、识别或 VIO。只保存各写入器准备好、共同录制起点之后的帧；已完成的 VIO 初始化和初始相对位姿会作为本次数据的起始状态保留。三维重建会额外保存录制起点的当前可见点云快照。**正在录制时，再按一次回车即可停止录制；相机和 Rerun 继续预览。** 文件在后台完成收尾，保存期间重复按回车不会重复操作。**保存和校验成功后自动弹出“Data Collector”面板**，显示“录制结束”和本次数据的实际保存位置。路径可选中复制，长路径自动换行；“打开文件夹”在系统文件管理器中打开该目录，“确定”、Esc 或关闭按钮仅关闭提示。提示尚未关闭时再次从 Rerun 或终端按回车会将提示带到前台；关闭后再按回车可打开设置面板录制下一份数据。每份使用独立文件夹，时间轴重新从本次录制开始。四个模式共用此行为；保存失败不会显示成功面板，错误和缓存位置仍输出到终端。无桌面或 `--headless` 时仅输出终端消息，退出整个程序时不弹出继续预览提示。

默认目录结构：

```text
/home/taoqiu/Ego2/output/
├── trajectory/时间戳/       # 单/多相机轨迹：LeRobot V3
├── gesture/时间戳/          # 手势识别：LeRobot V3
├── aprilgrid/时间戳/        # AprilGrid：Rerun 回放包
└── reconstruction/时间戳/   # 三维重建：Rerun 回放包
```

时间戳示例：`2026-10-30_16-30-25-123456`。手动录制下，`--session` 用于预填面板的名称和根目录，最终路径以面板预览为准。无界面的批处理可使用 `--headless`，它保留启动后立即录制的行为，`--session /绝对路径/新目录` 此时直接指定保存目录。旧数据不自动迁移；独立脚本仍保留原来的保存行为。

**轨迹和手势均为 LeRobot V3，`meta/info.json` 中 `codebase_version` 固定为 `v3.0`。** 一份数据集包含本次所有启用相机，根目录有 `session.json`、`meta/`、`data/`、`videos/`、`sensors/`、`calibration/`。Parquet、MP4 和 `meta/episodes/` 都按 V3 的 chunk/file 与 episode 索引组织。图像通道按部位和真实左右目命名，例如 `observation.images.head.cam1`（左目）、`observation.images.head.cam0`（右目）。SN 存在元数据中，不改变字段名。

- 轨迹：保存本地 VIO 位姿、公共坐标系位姿、有效性、地图编号、初始化状态、初始相对位姿及双目图像。位姿为以米为单位的行主序 4×4 变换矩阵，光学参考为 cam1，公共坐标系 Z 向上。原生后端将每次送入 VIO 并完成处理的帧写入独立归档（当前默认 15 Hz），不依赖会丢帧的预览环；接收到的有效 IMU 样本独立保存到 `sensors/<部位>/imu.parquet`。上游采集丢帧计数仍保留，不宣称是全速 USB 图像原始归档。
- 手势：保存双目 RGB、二维手部关键点、手势、置信度、推理有效性、原始模型结果与 IMU。原有 Bayer 保存开关继续生效，`--no-raw` 可关闭；启用时原始数据及帧索引在 `sensors/<部位>/raw/`。无推理的帧与“已推理但未检测到手”分别标记。
- 两种数据集都使用参考图像流的帧作为行：有胸部时取胸部，否则取第一个启用部位；其他相机使用时间最近的帧，容差为参考帧周期的 0.55 倍。`timestamp` 是 episode 内的定频视频时间，`observation.capture_time` 是共同接收时间轴；另存各设备原始时间戳、帧号、匹配误差、重复和缺帧标记。超出容差时使用黑色占位图并使位姿/手势无效，不使用旧位姿冒充测量。IMU 按原始频率独立保存，不随图像匹配而降采样。此配对不是硬件曝光同步，非参考流可能有未被选中的图像帧。
- AprilGrid 与三维重建最终只保留 `session.json` 和 `replay.rrd`。回放文件内包含画面、标注/定位结果或三维点云变化、时间轴与布局。三维重建运行时仍需要工作缓存，但 RRD 成功关闭并验证可读后会清理；该精简包不再保留旧导出流程所需的深度/关键帧缓存。

**停止录制：在前台 Rerun 窗口或启动终端按 Enter。退出整个程序：在启动终端按 Ctrl+C。** 两种操作都会完成尚未保存的录制；Enter 保留预览，Ctrl+C 同时停止相机。程序先完成各相机写入，再生成、检查 V3 数据集或 RRD，终端显示 `[saved]` 后才表示保存完成。结束阶段可能需要等待视频编码与校验。手动录制的 `session.json` 状态依次为 `preparing`、`recording`、`stopping`、`finalizing`、`complete`；失败为 `incomplete`，保留已产生的数据和工作缓存供检查。`--headless` 仍为立即录制并在退出时保存。关闭预览（`--no-viz`）不关闭保存；从未点击“开始录制”时，退出后清理预览临时缓存。Rerun 的播放/暂停只影响查看时间轴，不控制录制。

### 统一 Rerun 回放

执行 `/home/taoqiu/Ego2/run_multi_vio.sh`，切换到 **“数据回放”**：

1. 在左侧列表选择数据，或点击“浏览…”选择包含 `session.json` 的数据集文件夹；也可以直接粘贴路径。默认列表读取 `output/trajectory`、`output/gesture`、`output/aprilgrid`、`output/reconstruction` 下的数据，按创建时间倒序排列，支持刷新。
2. 查看右侧的功能模式、数据格式、保存状态、相机数量、录制时长及录制时的部位/SN。缺失的信息显示“未记录”；未选中、未保存完成或缺少必要文件时不能启动回放。
3. 点击“启动回放”。程序在后台检查文件、准备回放并启动 Rerun，成功启动后关闭 Sensia 面板；失败会在面板内说明原因。“取消准备”会停止后台任务并清理本次临时文件，返回选择状态。

面板底部提示为“无需连接相机，完全重现对应数据录制时的情况”。回放依据实际保存的数据，不重新运行识别、VIO 或重建算法；使用录制时的相机分配和布局，不依赖当前相机参数。录制开关只影响实时采集，在回放页不会开始新录制。Rerun 关闭后会清理本次 LeRobot 转换缓存，原始数据不变。

原有命令行回放仍可使用。四种模式自动从文件夹识别格式，将下面的路径替换为实际数据文件夹：

```bash
/home/taoqiu/Ego2/run_multi_vio.sh replay --session /home/taoqiu/Ego2/output/trajectory/2026-10-30_16-30-25-123456
```

LeRobot 数据集会转换成临时 RRD 后打开，支持拖动时间轴、暂停和调速，关闭查看器后清理临时回放文件；AprilGrid 和重建直接打开已保存的 RRD。回放只读取已保存的结果，布局恢复为对应模式，三维点云仍可旋转缩放。

需要保留一个可直接用 Rerun 打开的独立文件时：

```bash
/home/taoqiu/Ego2/run_multi_vio.sh replay \
  --session /home/taoqiu/Ego2/output/gesture/2026-10-30_16-30-25-123456 \
  --save /home/taoqiu/Ego2/output/gesture/2026-10-30_16-30-25-123456/replay.rrd \
  --headless
```

`--save` 不覆盖已有文件。录制中的保存和回放导出均使用有界内存；回放通过原生 Rerun 文件读取保留可跳转历史，而不是依赖实时窗口的环形内存缓存。

## 各功能启动命令

以下命令可在任意目录执行。单设备命令中的 `s200032` 可替换为已有配置的相机 SN；同一台相机一次只运行一个功能，按 `Ctrl+C` 结束。

### 1. AprilGrid 标定板跟踪

```bash
# 仅预览
/home/taoqiu/Ego2/run_aprilgrid.sh --sn s200032 --no-record

# 预览并保存数据
/home/taoqiu/Ego2/run_aprilgrid.sh --sn s200032
```

### 2. 手势识别

```bash
# 仅预览
/home/taoqiu/Ego2/run_gesture.sh --sn s200032 --no-record

# 预览并采集数据集
/home/taoqiu/Ego2/run_gesture.sh --sn s200032
```

### 3. 腕带标记位姿

首次使用或腕带几何改变后，先分别标定左右腕带，再启动位姿跟踪。

```bash
# 首次标定左右腕带
/home/taoqiu/Ego2/run_hand_pose.sh calibrate --hand left --sn s200032
/home/taoqiu/Ego2/run_hand_pose.sh calibrate --hand right --sn s200032

# 日常启动：跟踪并保存结果
/home/taoqiu/Ego2/run_hand_pose.sh --sn s200032
```

### 4. 三维重建

```bash
# 启动双目惯性 VIO、深度与点云重建，默认保存会话
/home/taoqiu/Ego2/run_reconstruction.sh --sn s200032
```

### 5. 胸部/头部/双手多相机 VIO 模式

启动后在面板中分配相机，支持 1～4 台。仅使用一台时，可分配到胸部、头部、左手或右手任意一个部位，其余部位留空；使用两台及以上时必须包含胸部。参考设备自动确定，配置中的 `reference_role: auto` 无需手动更改，`--no-setup` 和已保存的分配执行相同规则。

所有启用的相机均需先完成运动初始化，再观测同一块固定标定板完成对齐。单相机也沿用此流程，显示一条轨迹及一行双目画面，不显示相机间位姿差值区。坐标原点为初始标定板中心，方向取自参考设备完成初始化后的 VIO 坐标系，Z 轴向上。

```bash
# 弹出相机分配面板，仅预览
/home/taoqiu/Ego2/run_multi_vio.sh --mode vio --no-record

# 弹出相机分配面板，预览后按回车设置录制
/home/taoqiu/Ego2/run_multi_vio.sh --mode vio

# 使用上次保存的分配直接启动
/home/taoqiu/Ego2/run_multi_vio.sh --mode vio --no-setup --no-record
```

默认启用手动录制，启动后先预览；在前台 Rerun 窗口或启动终端按回车打开录制面板，再点击“开始录制”。再次按回车停止录制并保存，预览继续；保存完成后可录制下一份。画面、轨迹、初始相对位姿和标定快照保存为 LeRobot V3，默认目录为 `/home/taoqiu/Ego2/output/trajectory/创建时间/`。`session.json` 记录实际参考部位 `reference_role` 和 SN `reference_serial`，单相机的相机间相对位姿字段为空。在启动终端按 `Ctrl+C` 退出程序并完成尚未保存的录制。

## 项目结构

```text
Ego2/
├── README.md
├── run_aprilgrid.sh          # 标定板跟踪入口
├── run_gesture.sh            # 手势识别与采集入口
├── run_hand_pose.sh          # 腕带标记位姿入口
├── run_reconstruction.sh     # 三维重建入口
├── run_multi_vio.sh          # 四模式统一准备面板及启动入口
├── cameras/                 # 每台相机的采集配置及自身标定参数
│   ├── s200032/
│   ├── s200002/
│   ├── s200100/
│   ├── s200098/
│   └── s200097/
├── calibration_targets/     # 标定板与腕带标记参数
│   ├── calib.io_KALIBR_500x500_6x6_55.yaml
│   └── wrist_tags.yaml
├── configs/                 # 五个功能的运行配置及多相机分配记录
├── ego2/                    # 功能源码
│   ├── common/              # 共用输入、标定、几何及可视化基础代码
│   ├── aprilgrid/           # 标定板检测与跟踪
│   ├── gesture/             # 手势识别与数据集采集
│   ├── hand_pose/           # 腕带位姿与几何标定
│   ├── reconstruction/      # VIO、深度、点云及导出
│   └── multi_vio/           # 模式选择、采集调度、统一显示、VIO 对齐与保存
├── cyperstereo_isp_portable/ # 图像显示增强模块
├── models/                  # 手势识别模型
├── vocabulary/              # ORB 词袋
├── third_party/orb_slam3/    # ORB-SLAM3 及第三方源码
├── requirements*.txt        # Python 依赖清单
├── tests/                   # 测试与回归样本
├── docs/                    # 说明资源
├── build/                   # 本机编译产物
└── output/                  # 各功能生成的数据、轨迹与报告
```
