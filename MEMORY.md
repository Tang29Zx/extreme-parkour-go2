# MEMORY.md

## 架构与运行环境

- 本地源码路径：`/home/tang/extreme-parkour-go2`。
- Docker 启动脚本位于 `/home/tang/parkour/docker/extreme_go2/`。
- 本地 RTX 5070 Ti 与仓库固定的 PyTorch 2.4.1 CUDA 12.4 不兼容
  `sm_120` 执行，因此本地回放使用 CPU PhysX、CPU pipeline 和 CPU RL。
- Go2 预训练模型位于
  `legged_gym/logs/parkour_A100/607-02-16/`；当前能力基线使用
  `model_29500.pt`。

## 已验证踩坑

- 仓库根目录现已忽略 Python/测试/IDE/本地编译缓存以及新生成的训练日志；
  `legged_gym` 源码目录中误提交的 `__pycache__/*.pyc` 已解除 Git 跟踪。
  Isaac Gym `_bindings/linux-x86_64/py36` 下随 SDK 分发的 `.pyc` 保持跟踪，
  不能用全仓库批量删除缓存的方式处理。
- `play.py` 原本硬编码作者机器日志路径，已改为相对
  `LEGGED_GYM_ROOT_DIR`。
- `train.py` 原本启动后等待 debugpy，已移除该阻塞。
- `OnPolicyRunner` 原本不更新或恢复 `current_learning_iteration`，导致每次续训
  都从 `model_0.pt` 重新编号并在结束时覆盖它。现在新适配任务可从本地0开始，
  checkpoint记录完成轮数；后续从该checkpoint恢复时继续编号和保存。
- 单环境回放可使用 `--rows 1 --cols 1`；单行地形的课程难度固定为0，避免
  原公式 `i / (num_rows - 1)` 触发除零。
- 随机箱回放支持 `--random_box_layout 0～399` 精确选择固定seed布局；综合
  6箱数量、箱高、横向折返、短间距和强地面粗糙度，当前最复杂候选为130号。
- `rsl_rl/modules/estimator.py` 的无用 turtle 导入会要求 Tk，已移除。
- Docker 脚本使用 `EXTREME_GO2_IMAGE_NAME`，避免被其他项目的
  `IMAGE_NAME` 环境变量覆盖。
- 原生策略没有“最后一个障碍跳跃”或显式 jump action；jump schedule 调用
  处于注释状态。跳跃是 PPO 在孤立踏石、深坑和跨栏地形上，受朝目标方向
  速度与朝向奖励驱动后学出的行为。当前五箱 waypoint 设在箱体中心且没有
  箱间平地 waypoint，因此从第四箱看第五箱时会直接复用跨越动作。

## 当前任务决策

- 五箱能力测试保留 Extreme Parkour 原生132点扫描、Actor、PPO和奖励。
- 首版只迁移五箱地形与 waypoint，不迁移原 parkour 仓库的奖励、进度
  状态机、课程、KL或 checkpoint 操作器。
- 独立任务名为 `go2_five_box`；使用5个箱体中心和第五箱后平地作为
  6个顺序 waypoint，固定命令速度为 `0.5 m/s`。
- 已验证 `model_29500.pt` 可严格加载：4个CPU环境执行5步时观测形状
  为 `(4, 753)`、动作形状为 `(4, 12)`，观测和动作均为有限值。
- 能力回放中策略在原 `0.50 m` 第五箱稳定撞击；为单变量诊断，当前仅将
  第五箱降至 `0.40 m`，其他几何、扫描和命令保持不变。
- 后续观察确认失败发生在第四、第五箱之间，而非第五箱后的落地。原生目标
  推进只检查机身XY，不能据此认定已经稳定登上第五箱。默认第0环境的该段
  间距为 `1.40 m`，策略会强行直跳。短间距 `0.45～0.65 m` 会进一步匹配
  原生跳坑行为；当前诊断改用 `1.60～2.00 m` 的远间距，让第五箱在第四箱
  沿附近超出前向扫描范围，以测试策略能否先下地再接近。
- 独立 Sim-to-Real teacher 任务名为 `go2_random_box`：使用 `10×40=400`
  条固定种子布局，箱数2～6各80条，4096环境复用这些物理赛道；保留原生
  132点扫描、奖励、PPO和网络结构。该任务单独启用中等物理随机化、运行时
  Actor观测噪声及每局0～2控制步动作延迟，旧任务的新增开关默认关闭。
- 当前随机箱几何使用 `±1.0 m` 横向偏移和 `0.10～1.50 m` 均匀箱间距。
  400条赛道地面精确混合为30%平地、50% `0～0.02 m` 轻噪声和20%
  `0.02～0.06 m` 原仓库强噪声；类别固定seed打散。这些变化只作用于
  `go2_random_box`。
- `go2_random_box` 已用4个CPU环境严格加载 `model_29500.pt` 并连续执行5步，
  观测形状保持 `(4, 753)`，延迟0/1/2步均可采样；8环境检查确认摩擦、质量、
  质心和电机范围有效，噪声不产生NaN/Inf。相机噪声接口已配置，但teacher
  阶段仍使用高度扫描，深度渲染与蒸馏需在后续相机阶段单独验收。
- 几何适应任务 `go2_random_box_clean` 与完整随机箱任务共享几何和物理范围，
  但关闭Actor观测噪声、扫描偏移/抖动、接触丢失及动作延迟；用于从29500先
  训练300～500轮，再切换到完整Sim-to-Real随机化。
- 首次500轮 `random_box_clean_from29500` 是由修复checkpoint迭代记录之前的
  服务器进程生成；`model_0/100/200/300/400.pt`内部均记录iteration 0，
  其中修改时间最晚的`model_0.pt`是训练结束权重。切换到新run时从该文件加载，
  新版runner会重新从0正确编号。
- 400条 `25×4 m` 随机箱赛道若使用 `0.05 m` grid mesh，会生成约1768万
  顶点和3534万三角形，并在服务器 `gym.add_triangle_mesh()` 时触发段错误。
  两个随机箱任务因此改用仓库已有的 Delatin fast mesh，最大几何误差为
  `0.01 m`；原生任务继续使用原来的 grid mesh。

## 真机部署约束

- `model_29500.pt` 是使用仿真地形高度扫描点的 teacher policy，不能把该
  checkpoint 原样部署到 Go2；真机没有 Isaac Gym 的 heightfield 查询。
- 推荐路径是先用最终 teacher policy 蒸馏得到深度相机 student，再导出匹配的
  base JIT 与 vision weights。teacher policy、视觉编码器与导出权重必须来自
  同一训练版本，不能把旧视觉权重和后续修改过的 Actor 混用。
- 当前仓库有视觉蒸馏与 JIT 导出代码，但不包含完整、已验证的 Go2 真机低层
  控制程序；真机侧仍需适配 Unitree SDK2、D435i 深度输入、观测顺序、关节
  顺序/符号、动作缩放、默认姿态、PD 参数、控制频率、超时 watchdog 和急停。
- 上真机顺序必须是离线权重等价检查、只读 dry-run、悬挂/支架检查、平地低速
  测试、低矮单障碍测试，最后才是五箱；低层控制前必须避免与原生 sport mode
  同时发命令。
