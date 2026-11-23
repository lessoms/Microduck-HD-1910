# Microduck-HD-1910

## Radxa ZERO 3W 联合模型部署

六种动作的独立 CPU 推理部署包见 [radxa/README.md](radxa/README.md)。[下载完整模型与运行包](https://github.com/lessoms/Microduck-HD-1910/releases/tag/radxa-v1.0.0)，解压后执行 `bash setup.sh`，再运行 `bash run.sh --offline --seconds 3`。无需训练工程、MuJoCo 或 CUDA。

下载后可直接安装并运行模拟硬件测试；真实鸭子运动仍需配置舵机和躯干 IMU 标定、核验板端实时性，再显式启用输出。发布与维护：lessoms。

## 已训练动作与统一仿真控制

完整模型、运行脚本、机器人资源及训练历程见 [simple_movements](simple_movements/README.md)。

```bash
cd simple_movements
bash setup.sh
bash run.sh
```

## 动作演示（MuJoCo 仿真）

以下展示最终模型的行走、转弯、倒退、坐下起立和前滚翻能力。画面为 MuJoCo 仿真，GIF 下方可查看 MP4 原视频。

| 行走 | 转弯 |
| --- | --- |
| **行走**<br>![行走：MuJoCo 实际策略运行](simple_movements/media/gifs/walking.gif)<br>[MP4 原视频](simple_movements/media/videos/walking.mp4) | **转弯**<br>![转弯：MuJoCo 实际策略运行](simple_movements/media/gifs/turning.gif)<br>[MP4 原视频](simple_movements/media/videos/turning.mp4) |
| 倒退 | 坐下与起立 |
| **倒退**<br>![倒退：MuJoCo 实际策略运行](simple_movements/media/gifs/backward.gif)<br>[MP4 原视频](simple_movements/media/videos/backward.mp4) | **坐下与起立**<br>![坐下与起立：MuJoCo 实际策略运行](simple_movements/media/gifs/sitstand.gif)<br>[MP4 原视频](simple_movements/media/videos/sitstand.mp4) |
| 前滚翻与起身 | |
| **前滚翻与起身**<br>![前滚翻与起身：MuJoCo 实际策略运行](simple_movements/media/gifs/roulade.gif)<br>[MP4 原视频](simple_movements/media/videos/roulade.mp4) | |

[完整训练与实现历程](simple_movements/README.md)

## 进阶动作

右脚踢球、嘴部夹取和单脚站立的策略权重、MuJoCo 运行入口与演示视频见 [advanced_movements](advanced_movements/README.md)。

```bash
bash advanced_movements/run.sh
```
