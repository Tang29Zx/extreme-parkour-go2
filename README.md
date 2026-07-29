# Unitree Go2 上的极限跑酷代码

这个仓库在**Unitree Go2**四足机器人上部署[Extreme Parkour with Legged Robots](https://github.com/chengxuxin/extreme-parkour)项目。原始工作基于A1机器人开发。

## Onboard安全边界单箱评测

`legged_gym/legged_gym/scripts/evaluate_onboard_single_box.py`将本仓库的固定单箱、
Sim-to-Real随机化和Isaac深度相机，与`Extreme-Parkour-Onboard`生产端同源的
LowState/LowCmd及输出保护串成闭环。运行容器必须同时挂载两个仓库；本机RTX 50系
使用CPU PhysX、CPU pipeline和CPU相机回读：

```bash
python legged_gym/legged_gym/scripts/evaluate_onboard_single_box.py \
  --task go2_random_box --headless \
  --device cpu --pipeline cpu --rl_device cpu --graphics_device_id 0 \
  --use_camera --cpu_camera --num_envs 20 --seed 17 \
  --fixed_box_height 0.25 --fixed_ground_roughness 0.005 \
  --policy_duration_s 15 \
  --jit_model_dir /workspace/Extreme-Parkour-Onboard/traced \
  --onboard_root /workspace/Extreme-Parkour-Onboard \
  --episode_output artifacts/onboard_box025.json
```

将`--fixed_box_height`改为`0.45`可评测高箱。输出JSON分别统计模型动作裁剪、接入
保护、持续目标步长、机械关节限位和预测PD力矩边界；成功必须完成箱体中心和箱后
出口两个waypoint。该结果是仿真统计，不构成真机接管授权。

# 致谢

这个仓库基于对[Robot Parkour Learning](https://github.com/ZiwenZhuang/parkour)、[Extreme-Parkour-Onboard](https://github.com/change-every/Extreme-Parkour-Onboard)的修改。在此特别感谢原始作者的开源贡献。
