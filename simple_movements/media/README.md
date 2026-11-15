# 动作演示录制

五个片段使用最终部署包的三个actor，在同一配置的MuJoCo物理环境中运行。固定种子20261111，每段独立创建控制器，未改动策略权重、限幅、执行器参数或技能交接逻辑。每段只有初始一次reset；记录结束后均确认回到standing。

- 行走：名义vx=+0.18、wz=0，运动5秒并记录2秒停车。
- 转弯：名义vx=0、wz=+0.75，左转5秒并记录2秒停车。
- 倒退：名义vx=-0.18、wz=0，运动5秒并记录2秒停车。
- 坐下起立：通过控制器S/U对应命令，保留坐姿停留和恢复站立。
- 前滚翻：通过控制器F对应命令，保留站姿准备、翻滚、起身和停稳。

名义指令与实际物理速度的区别见主README。演示表示这些种子/流程下的实际行为，不作为新的批量成功率评估。

## 文件

| 动作 | 时长 | GIF | 原视频 | 物理运行记录 |
| --- | ---: | --- | --- | --- |
| 行走 | 7.0秒 | [GIF](gifs/walking.gif) | [MP4](videos/walking.mp4) | [报告](walking_report.json) |
| 转弯 | 7.0秒 | [GIF](gifs/turning.gif) | [MP4](videos/turning.mp4) | [报告](turning_report.json) |
| 倒退 | 7.0秒 | [GIF](gifs/backward.gif) | [MP4](videos/backward.mp4) | [报告](backward_report.json) |
| 坐下与起立 | 9.8秒 | [GIF](gifs/sitstand.gif) | [MP4](videos/sitstand.mp4) | [报告](sitstand_report.json) |
| 前滚翻与起身 | 10.8秒 | [GIF](gifs/roulade.gif) | [MP4](videos/roulade.mp4) | [报告](roulade_report.json) |

## 画面如何生成

控制器以50Hz真实推进物理，按25fps记录实际qpos，随后用相同MuJoCo模型离屏渲染。相机仅跟随位置、保持固定角度；不重写机器人姿态、不插值动作、不替换为教师轨迹。视频添加标题、时间与仿真标记。GIF由同一MP4转换为480×360、20fps，保持原时长并循环播放。播放器循环回到片头不属于动作中途reset。

```bash
cd simple_movements
bash setup.sh
MUJOCO_GL=egl bash record_demos.sh
# 单独重录一个动作：
MUJOCO_GL=egl bash record_demos.sh --only roulade
# 保留已有片段，继续缺失部分：
MUJOCO_GL=egl bash record_demos.sh --skip-existing
```

录制入口为 `runtime/record_demos.py`；GIF位于gifs/、原视频位于videos/。报告保存事件、采样速度、模型manifest哈希、GIF哈希与实际帧数。渲染预览图写入ignored的outputs/目录。
