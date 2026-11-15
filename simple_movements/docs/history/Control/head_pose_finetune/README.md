# 头姿独立微调实验

保留原400轮与Control统一控制器，使用独立目录进行教师头姿统计、软约束、critic校准、10轮短微调及完整起停对照。没有锁死关节或在播放时覆盖头部输出，原模型文件未修改。

## 教师目标与奖励

统计原63条完整教师轨迹共37,800步的4个头颈编码器角度（电机角+背隙角）。目标取初始站立0.5–2秒与最终10–12秒的中位数；活动死区取全流程相对目标偏差99分位+2°，并设置保守下限，允许运动中的自然摆头。不是将所有角度设0。

|关节|目标角°|允许偏差|位置惩罚权重|
|---|---:|---:|---:|
|neck_pitch|8.92|±20.00|0.025|
|head_pitch|22.11|±20.00|0.025|
|head_yaw|0.36|±9.27|0.12|
|head_roll|1.91|±8.55|0.12|

```text
excess = max(abs(encoder_angle-target)-deadband, 0)
position_cost = Σ weight * (sqrt(1+(excess/scale)²)-1)
velocity_cost = Σ velocity_weight * (sqrt(1+(encoder_velocity/1rad/s)²)-1)
new_reward = original_R8_reward - position_cost - velocity_cost
```

位置尺度：颈/头pitch30°，yaw/roll10°。速度权重分别0.002/0.002/0.01/0.01。主要限制侧倾和偏转，俯仰保留较宽范围。死区内位置项为0，越界后平滑增加；梯度有限且指向回正，未引入动作硬限位。训练进程内使用同一函数计算真实环境、离线/critic和想象的头姿代价，现有Control文件没有被改成新奖励。

## 已执行的训练方法

1. 加载 `velocity_turn_original/full_flow_r7/progress_r8/joint_400/checkpoint_0400.pt`，校验SHA256为 `83b9ec0fb902ccc10bb3394a650891fd7796ad304e8a48d9a05300237b6c22ee`。
2. 原400轮真实42场景复测：完整42/42；记录站立、运动、停车阶段头姿。原400策略、世界模型与优化器状态均加载，策略学习率降至0.000025（原值四分之一）。
3. critic额外校准1,000次：500次原64k物理回放，500次400轮保存的真实学生在线回放，交替更新critic/target。学生回放按episode/age排序，使用原pose_history并重新计算新奖励；不复用旧reward值。校准后核对world和policy权重逐张量不变。保留原在线buffer、计数、困难采样器与next_episode_id。
4. 保存新的400轮迁移副本到prepared目录，从其继续401–410轮，沿用原SuperMimic联合优化、原网络、32步想象、GAE、target critic及随机化。没有PPO、在线学生教师纠偏或重新从0训练。
5. 新collection包含头姿配置及哈希，版本head_pose_soft_v1，旧400轮不能直接被当作同目标续训，需要显式迁移。prepared/calibration.json记录校准和不变性检查。

## 短训练结果

|指标|原400轮复测|头姿微调410轮|
|---|---:|---:|
|原完整起停测试|42/42|42/42|
|附加请求运动轴相对误差≤25%|31/42|35/42|
|头yaw全流程平均绝对目标偏差|34.97°|4.10°|
|头roll全流程平均绝对目标偏差|15.64°|9.41°|

偏转改善88.3%，侧倾改善39.8%。初始站立时头roll均值约17.66°→8.99°，头yaw均值约-12.90°→+0.42°。仍有剩余侧倾，尤其运动/停车阶段，不代表头部始终完全端正。俯仰也接近教师范围，但不是锁定固定角度。

采用候选门槛：完整42场景成绩不降、附加25%子集数不降，且yaw/roll平均绝对目标偏差各下降至少10%。410轮通过这组固定测试门槛；尚不能据此宣称任意种子或实机可靠。原400轮此前正式日志附加25%子集为33，本次新复测为31，存在浮点动力学/复测波动，因此使用本次成对基线31进行比较，不覆盖原成绩。

奖励回归通过：教师范围内位置惩罚为0，越界代价正负对称，有限梯度朝回正方向。记录见reward_validation.json。基线与候选JSON包括存活样本数；失败时的低头姿统计不能作为改善依据，候选必须同时通过运动门槛。

## 文件

- config.json / build_config.py：教师范围与小权重配置。
- task.py：仅当前Python进程使用的头姿奖励适配器，现有Control速度任务不变。
- train.py / run.sh：显式迁移校准与短训练入口，防止误覆盖已有输出。
- baseline400.json：重新测得的原400轮42场景与头姿。
- prepared/checkpoint_0400_head.pt：新目标下校准critic后的起点，非旧模型覆盖。
- short_train/checkpoint_0410.pt：10轮短训练候选。
- short_train/evaluations.jsonl、comparison.json、RESULTS.md：原始评估、成对比较与结论。
- evaluate.py：独立复测，支持指定新随机种子。

当前统一控制器仍默认原400模型，未自动部署410。若人工检查候选，可打开独立窗口：

```bash
bash /home/zhichao/supermimic/Control/head_pose_finetune/view.sh
```

沿用速度窗口W/S前后、A/D转向、空格归零、R重置、Q退出。此窗口与统一技能控制器的W/X键位不同，以终端提示为准。

## 复现与继续

已经完成prepare与10轮train，无需重复执行。新实验副本的初始操作是：

```bash
bash /home/zhichao/supermimic/Control/head_pose_finetune/run.sh prepare
bash /home/zhichao/supermimic/Control/head_pose_finetune/run.sh train --rounds 410
```

已有输出会拒绝覆盖。若以后明确继续同一个配置，可显式续训：

```bash
bash /home/zhichao/supermimic/Control/head_pose_finetune/run.sh train --rounds 420 \
  --resume-checkpoint /home/zhichao/supermimic/Control/head_pose_finetune/short_train/checkpoint_0410.pt
```

配置改变必须新建实验副本、重新标记目标并校准critic，不能直接将旧head checkpoint认作同一奖励。本次仅到410，未启动上述继续命令。


## 额外随机种子复测

seed20261128下重新运行原400与新410，各42场景均完整通过42/42；头yaw平均绝对目标偏差约35.15°→4.06°，roll约15.66°→9.38°。附加25%子集为31→35。原始记录baseline_seed20261128.json与candidate_seed20261128.json，汇总extra_seed_comparison.json。该结果仅增加一组成对证据，不能代表所有随机环境；步态中的侧倾峰值仍约20°，头部侧倾尚未完全消除。


## 头姿第二轮调整

已在head_pose_finetune_v2从410轮做411–420轮短微调：零指令安静范围收紧为yaw±4°/roll±3°，运动保留自然摆动范围。固定42场景仍42/42；站立roll均值约8.98°→4.03°，停车后11.28°→5.68°；全流程roll平均绝对偏差9.39°→5.73°。当前统一控制器未替换。详见[第二轮说明](/home/zhichao/supermimic/Control/head_pose_finetune_v2/README.md)。
