# Radxa ZERO 3W 联合模型部署包

维护与发布：**lessoms**。

针对 RK3566 的独立 CPU 推理软件：速度控制、坐下/起立、前滚翻、右脚踢球、嘴部夹取、右脚支撑单脚站立。模型约 200 MiB，保留原 actor 权重，不重新训练；单脚站立保留落脚子策略。模型属于多策略组合包，不是重新训练的共享神经网络。

**下载后可安装并直接运行模拟硬件测试。真实鸭子运动前仍需完成本机标定与性能验收，不能免标定直接运动。** 默认不写舵机目标，不自动开扭矩。当前未在 Radxa 实板上验证。

## 1. 推荐：下载完整包

在 Radxa 的终端或 SSH 会话执行：

```bash
mkdir -p ~/microduck
cd ~/microduck
curl -fL https://github.com/lessoms/Microduck-HD-1910/releases/download/radxa-v1.0.0/microduck-radxa-v1.0.0.tar.gz -o microduck-radxa-v1.0.0.tar.gz
curl -fL https://github.com/lessoms/Microduck-HD-1910/releases/download/radxa-v1.0.0/SHA256SUMS -o SHA256SUMS
sha256sum --ignore-missing -c SHA256SUMS
tar -xzf microduck-radxa-v1.0.0.tar.gz
cd microduck-radxa
bash setup.sh
bash run.sh --offline --seconds 3
```

完整压缩包包含联合模型、运行源码、参考特征、配置模板、安装脚本及许可证。无需原训练工程、MuJoCo、CUDA、相机或足底传感器；安装依赖需要网络。成功结束会报告软件周期统计。

## 2. 或从 GitHub 源码安装

只检出 Radxa 所需目录，不下载仿真和训练资产：

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/lessoms/Microduck-HD-1910.git
cd Microduck-HD-1910
git sparse-checkout set radxa
cd radxa
bash setup.sh
bash run.sh --offline --seconds 3
```

setup.sh 自动下载 Release 中的 combined_model.pt，校验文件大小及 SHA-256，建立 .venv 并安装 CPU 推理依赖，再运行控制测试。模型不放在普通 Git 文件中，避免 GitHub 单文件大小限制；无需 Git LFS。

系统要求：64 位 Linux（Radxa 用 aarch64）、Python 3.10 或更高。建议先确认 `python3 --version` 和 `uname -m`。若系统没有对应的 CPU PyTorch 二进制包，安装会明确失败，需更换兼容的系统/Python；脚本不会在板子上编译 PyTorch。Debian 缺少 venv 时先安装与 Python 对应的 python3-venv。不能复制电脑的 x86 虚拟环境到 ARM 板。

若已下载完整包且只想还原配置、验证模型，不安装 Python 依赖：

```bash
bash setup.sh --prepare-only
```

## 3. 接电脑现有网页

电脑继续运行你的 Control_Platform（不是本仓库仿真 run.sh）：

```bash
bash /home/zhichao/supermimic/Control_Platform/run.sh
```

电脑另一个终端建立 SSH 反向隧道，将实际账号和 IP 替换进去：

```bash
ssh -N -R 18080:127.0.0.1:8080 用户名@Radxa的IP
```

Radxa 上运行：

```bash
cd ~/microduck/microduck-radxa
bash run.sh --hardware mock --url ws://127.0.0.1:18080/ws
```

电脑浏览器打开 http://127.0.0.1:8080 ，点击连接。该模式明确标为模拟硬件；其位置跟随不是物理仿真，不代表动作成功。启动前关闭 sender.html 发布端，因为代理只允许一个 producer。若代理设置 CONTROL_TOKEN，板端进程设置相同环境变量。

### 动作按键

| 按键 | 请求 |
| --- | --- |
| S / 1 | 坐下 |
| U / 2 | 起立 |
| F / 3 | 前滚翻 |
| K / 4 | 踢球 |
| G / 5 | 夹取 |
| B / 6 | 单脚站立 |
| W / X | 前进 / 后退 |
| A / D | 左转 / 右转 |
| 空格 | stop 请求 |
| Esc | 急停请求 |

按键表对应已适配的电脑网页；原版网页若未绑定进阶按键，仍可使用其动作按钮。行走按住续发、松开归零。没有单独训练停车模型，stop 只将速度目标归零；不代表实测停稳。单次动作收到显式 stop 会锁存保持；代理的 100 ms broker_watchdog 不误中断单次动作。

## 4. 只读硬件识别与标定

硬件：Radxa + Microduck HAT + 飞特 HD1910；HAT 安装在头部，外接躯干 IMU 用于模型输入。

```bash
.venv/bin/python diagnose.py
# 确认串口和实际 ID 后，才使用实际 ID 替换下面示例。
.venv/bin/python diagnose.py --port /dev/ttyS2 --ids 11 12
```

诊断不写目标、不写 EEPROM、不启用扭矩。避免与 robotd/Studio 等进程同时占用舵机总线。/dev/ttyS2 和 1 Mbps 来自参考项目记录，不是你的硬件已经验证的参数。

setup.sh 从 config.example.json 创建 config.json；再次安装不会覆盖本地 config.json。后者被 Git 忽略，勿把你的私人配置和凭据提交到仓库。

需要在 config.json 填写并核实：

- 14 个模型关节的 ID、零位 tick、正反方向、ticks_per_rad、速度反馈单位和机械范围，逐项 verified=true。
- body_imu 的 Linux IIO 加速度/陀螺仪节点、sensor_to_body 安装旋转矩阵、静止陀螺偏置，标定后 mount_verified=true。
- 若启用嘴部，填写第 15 路舵机映射、角度范围、实际测得的电流保护阈值及 verified。不要用其他型号或仿真扭矩值代替实测电流阈值。

IMU 接口要求系统已识别 LSM6 芯片，暴露 in_accel_[xyz]_raw、in_accel_scale、in_anglvel_[xyz]_raw、in_anglvel_scale；加速度和陀螺可以位于不同 IIO 设备。未包含内核/设备树驱动安装，不把头部 IMU 当躯干输入。

完成配置后先运行不写目标的监测模式：

```bash
bash run.sh --hardware feetech --url ws://127.0.0.1:18080/ws
```

核实输入、映射、准备姿态和板端时序后，才将 real_motion_verified 设为 true，并显式启用输出：

```bash
bash run.sh --hardware feetech --enable-write --url ws://127.0.0.1:18080/ws
```

程序不会自动开扭矩、刷固件、修改 PID 或 EEPROM。首版只有位置目标输出，不提供在线标定向导。

## 5. 控制行为与限制

- 按动作键代表人工确认；程序另检查十帧观测、IMU 稳定性和准备关节姿态。切换动作不复位实体、不伪造历史。不满足准备姿态时拒绝执行，不自动从任意姿态准备动作。
- 50 Hz 为目标控制频率；必须实测板端推理加总线读写能否在 20 ms 内完成。当前报告中的电脑软件测试不等于 Radxa 实时性能。
- 进阶动作和前滚翻使用阶段参考姿态容差，允许正常倾斜；结束后保持末目标并等待你确认，不声称触球、夹取或站稳成功。
- 急停、断连、异常姿态或周期超时锁存保持；保持目标不是物理断电，也不是可靠的跌倒恢复。实际急停机械行为需单独验证，锁存后本地检查并重启。
- 嘴部时序与电流保护接口已接入；未实机复现仿真的 5 ms 扭矩保护与保持力矩。默认禁止未经标定的真实夹取。
- 没有足底传感器，不自动确认承重。未包含原仿真基础控制器的全部步态交接和姿态恢复逻辑。
- 头部 HAT IMU、WebRTC 相机及 ToF 发布尚未接入本运行程序。网页躯干四元数为重力倾斜估计，未测绝对航向。
- 暂拒绝网页单舵机手动请求，避免与模型争用。没有 NPU/RKNN 后端或量化；当前是 CPU FP32。

## 文件与验证

run.py / controller.py / policy.py：运行入口、动作管理、独立 actor 推理。hardware.py / feetech_bus.py：硬件接口。assets/：部署所需参考数据。diagnose.py：只读诊断。

```bash
.venv/bin/python test_deployment.py
```

六套 actor 及单脚落脚子策略曾与原实现做逐元素输出比对；控制测试、模拟硬件和电脑网页 WebSocket 链路测试已通过。详见 validation.json。真实硬件和实体动作尚未验收。

飞特通信代码来自 [fanhao375/microduck-replica](https://github.com/fanhao375/microduck-replica)，保留 Apache-2.0 许可证与署名，见 NOTICE、LICENSE-APACHE-2.0 和 UPSTREAM_LICENSE。
