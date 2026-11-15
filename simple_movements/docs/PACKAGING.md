# 发布包与原工作区的对应关系

本目录只打包最终连续仿真控制需要的模型和代码。模型选择为速度420、坐站旧第二次120、前滚翻新380；原工作区README完整保存在 `history/README.md`，其直接引用的历史资料按原相对结构归档。归档内更早的链接和命令仍可能引用未发布的训练数据，不能用作本包运行入口。

## 打包时做的适配

- 从三个完整训练checkpoint导出actor和collection_config。actor所有tensor逐项相等，固定随机输入前向输出最大绝对差为0，详见 `../models/manifest.json`。不包含世界模型、critic、优化器、训练replay和随机数状态，不支持直接恢复完整训练。
- 将运行引用改为本目录的models/references路径，移除原机器绝对路径依赖。参考轨迹保持原文件字节，包含原执行器/延迟/齿隙快照。
- 控制器保留原物理步进、限幅、状态判断和动作渐变；只修改import和默认模型路径。
- 从old_aligned.py提取原actor定义，从velocity_joint_recipe.py提取原condition指令编码，避免播放时导入训练器、旧240轮checkpoint、标定文件或教师ONNX。
- 从原录制/诊断脚本提取环境配置，避免为了播放导入录像与教师推理工具。域随机化、模型参数和仿真环境实现保持原有行为。
- vendor保留实际训练所用mjlab_microduck源码与机器人资源，包括Feetech初始化修复；不依赖原工作区editable安装。
- 运行脚本按脚本自身位置定位目录，支持SIMPLE_MOVEMENTS_PYTHON指定兼容解释器；默认使用setup.sh创建的独立.venv。
- 固定mjlab、MuJoCo、Warp、PyTorch及实际BAM源码提交；uv.lock锁定完整解析结果。GPU首次运行需要编译Warp内核，通常比再次运行慢。

## 验证入口

```bash
# 文件哈希、推理tensor/形状、参考路径、导入来源
bash verify.sh
# 原坐站→行走→停车→翻滚组合
MUJOCO_GL=egl bash run.sh --headless --demo-combined --max-seconds 90 --report outputs/combined.json
# 运动中请求坐下、转向中请求翻滚
MUJOCO_GL=egl bash test_handoffs.sh
```

`validation/source_*.json`是原工作区既有回归；packaged_*.json是发布目录实测，二者分开记录。测试只覆盖给定种子和动作流程，交互GUI需在有图形显示的桌面运行。
