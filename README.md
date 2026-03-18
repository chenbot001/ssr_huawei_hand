# SSR Huawei 遥操作与硬件控制项目

本项目实现了一套基于 GELLO 的遥操作系统，用于控制 UR5 机械臂和睿研灵巧手，并集成了多路视觉传感器（RealSense 和指尖触觉相机）。

## 📁 项目结构

```
.
├── configs/                # 配置文件 (硬件参数, 遥操作参数)
├── scripts/                # 启动脚本与测试工具
├── src/
│   └── ssr/                # 核心源码包
│       ├── control/        # 控制器逻辑 (TeleopController)
│       ├── hardware/       # 硬件驱动 (UR5, GELLO, Ruiyan Hand, Cameras)
│       └── utils/          # 工具函数 (可视化, 相机设备管理)
├── external/               # 外部依赖 (gello_software 等)
└── pyproject.toml          # Python 项目配置
```

## 🛠️ 环境安装

本项目依赖于 Anaconda 环境 `ssr_huawei`。

1. **激活环境**
   ```bash
   conda activate ssr_huawei
   ```

2. **安装项目依赖**
   在项目根目录下运行，以安装 `ssr` 包及其依赖（Editable 模式）：
   ```bash
   pip install -e .
   ```

## ⚙️ 硬件配置

所有的硬件连接配置均位于 `configs/hardware_config.yaml`。本系统采用了**基于 USB ID 的动态相机发现机制**，解决了 `/dev/video*` 设备号重启后漂移的问题。

### 关键配置项
*   **UR5**: 设置 IP 地址 (默认 `192.168.1.5`)。
*   **GELLO**: 设置串口路径 (如 `/dev/serial/by-id/...`)。
*   **Ruiyan Hand**: 使用 SocketCAN 接口 (默认 `can0`)。
*   **Cameras**: 使用 `v4l2-ctl` 获取的稳定 USB ID。

### 获取相机 ID
如果更换了 USB 插口，请运行以下命令查看新的设备 ID：
```bash
v4l2-ctl --list-devices
```
找到类似 `usb-0000:00:14.0-5.2` 的字符串，并更新到 `configs/hardware_config.yaml` 中。

## 🚀 使用指南

### 1. 硬件连接检查 (推荐)
在运行主程序前，建议先运行连接测试脚本，确保所有硬件（机械臂、灵巧手、相机）均已连接并响应。
```bash
python scripts/test_connection.py
```
如果输出 `ALL SYSTEMS GO`，则表示硬件状态正常。

### 2. 启动遥操作
启动主遥操作程序：
```bash
python scripts/run_teleop.py
```

### 3. 组件独立测试
如果遇到问题，可以使用以下脚本单独测试各个组件：
*   **机械臂**: `python scripts/test_ur5.py`
*   **灵巧手**: `python scripts/test_hand.py`
*   **GELLO**: `python scripts/test_gello.py`
*   **指尖视触觉**: `python scripts/test_tactile.py` 
*   **RealSense**: `python scripts/test_realsense.py`

## 🔧 故障排除

*   **CAN 通讯失败**: 请运行初始化脚本启动 CAN 接口：
    ```bash
    source scripts/ryhand_init.sh
    ```
    或者手动执行命令确保接口已启动 (`sudo ip link set can0 up type can bitrate 1000000`)。
*   **相机未找到**: 检查 `configs/hardware_config.yaml` 中的 `id` 是否与 `v4l2-ctl --list-devices` 输出一致。
*   **UR5 连接超时**: 检查网线连接及本机 IP 设置是否与机械臂在同一网段。

## 📝 开发者说明

*   核心硬件接口类位于 `src/ssr/hardware/`。
*   相机索引通过 `src/ssr/utils/camera_utils.py` 动态解析。
*   所有的硬件驱动命名已统一为 `RyHandDriver`, `GelloController`, `UR5Arm` 等。

---

# GELLO 设置指南

本指南提供了设置和运行 GELLO 的说明。

## 1. 安装依赖

如果尚未配置 GELLO 环境，请安装其依赖项：

```bash
cd external/gello_software
pip install -r requirements.txt
pip install -e .
pip install -e third_party/DynamixelSDK/python
cd ../..
```

## 2. 关节偏移校准

运行校准脚本以确定 GELLO 设备的零位偏移量。

```bash
python external/gello_software/scripts/gello_get_offset.py \
  --start-joints 1.57 -1.57 1.57 -1.57 -1.57 0 \
  --joint-signs 1 1 -1 1 1 1 \
  --port "/dev/serial/by-id/..."
```

> **注意：** 
> 1. 请将 `--port` 替换为您实际的 GELLO 串口路径。
> 2. `--start-joints` 参数应填写**机械臂**（从动端）当前的真实关节角度（弧度制）。
> 3. 运行脚本后，记录下输出的 `offset` 数组。

## 3. 配置偏移量

打开文件 `external/gello_software/gello/agents/gello_agent.py`，找到 `PORT_CONFIG_MAP` 字典，更新您设备的偏移量配置。

## 4. 注意事项

本项目通过 `configs/hardware_config.yaml` 管理 GELLO 的串口号。请确保配置文件中的 `gello.port` 正确。


