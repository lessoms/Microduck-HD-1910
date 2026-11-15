# 来源和第三方许可

- 机器人/执行器训练包来自本地Microduck replica训练快照，基础来源为Pollen Robotics/microduck_rl与fanhao375/microduck-replica。代码原有Apache-2.0许可见TRAINING_APACHE_2.0.txt，复刻仓库许可说明见REPLICA_LICENSE.txt。
- Microduck三维模型与网格保留Pollen Robotics的CC BY-NC-SA 4.0声明；代码Apache许可不将这些网格改为Apache。对应上游：https://github.com/pollen-robotics/microduck_rl 、https://github.com/fanhao375/microduck-replica 。
- HD-1910参数来自LuwuDynamics/xgoduck_rl，Apache-2.0；原参数来源说明和许可证保留在vendor/mjlab_microduck/robot/hd1910/。
- BAM由Rhoban提供，依赖固定到62bd8ce12154340be97e06f7f41a0ca8f116d967；其许可证随依赖安装：https://github.com/Rhoban/bam 。
- 三个actor由此工作区训练产物原样导出，模型来源与校验值见models/manifest.json；未将教师ONNX、账户配置或训练日志作为运行依赖上传。

本清单用于保留来源及第三方声明，不将所有不同来源的资源统一重新许可。
