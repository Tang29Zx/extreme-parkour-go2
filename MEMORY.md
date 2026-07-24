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

- 原仓库基线提交内共跟踪149个`.pt`文件，约1.5GB，其中140个是训练
  checkpoint。主要有两套不同的teacher序列：
  `logs/extreme_parkour_a1_test/607-00-06`（87个checkpoint）和
  `logs/parkour_A100/607-02-16`（49个checkpoint）；两处
  `model_29500.pt`哈希不同，不能混用。多个目录中的`base_jit.pt`和
  `vision_weight.pt`内容完全相同，只是重复副本。
- 原仓库自带的`base_jit.pt`与`vision_weight.pt`是Go2视觉推理产物：
  配置使用Go2 URDF、53维本体观测、12维动作、87×58深度图、10步历史、
  `action_scale=0.25`和PD参数`Kp=40/Kd=1`。它们只能成对用于推理，
  不能恢复PPO训练；重复副本旁的相机配置并不完全一致，来源关系不够可靠，
  因此只能作为仿真基线，不能未经等价性和安全验证直接部署到真机。
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
- `random_box_sim2real_from30000` 已生成编号正确的30100～30500 checkpoint；
  全部模型及优化器张量有限。30500相对30000的Actor参数相对变化约3.0%，
  自适应学习率最终为`1e-5`。当前runner的TensorBoard writer被注释，使用
  `--no_wandb`时目录只保存checkpoint，不保存奖励、通过率或终止趋势；后续
  训练若需离线分析必须保留终端输出或恢复本地指标日志。
- 标准PPO和视觉蒸馏均固定每`50`个iteration保存一次checkpoint；续训时以
  本次恢复后的相对iteration计数触发，最终退出轮若不在50倍数上仍会额外保存。
- 400条 `25×4 m` 随机箱赛道若使用 `0.05 m` grid mesh，会生成约1768万
  顶点和3534万三角形，并在服务器 `gym.add_triangle_mesh()` 时触发段错误。
  两个随机箱任务因此改用仓库已有的 Delatin fast mesh，最大几何误差为
  `0.01 m`；原生任务继续使用原来的 grid mesh。

## 真机部署约束

- `model_29500.pt` 是使用仿真地形高度扫描点的 teacher policy，不能把该
  checkpoint 原样部署到 Go2；真机没有 Isaac Gym 的 heightfield 查询。
- 随机箱视觉蒸馏从noisy teacher `model_30500.pt`启动；相机模式默认使用
  192环境，每轮采集120步。首次加载时depth actor复制teacher Actor，teacher
  冻结，深度编码器和depth actor通过动作模仿与yaw损失更新。当前代码中的
  独立`depth_encoder_loss`被注释，因此显示0属预期；应监控depth actor loss、
  yaw loss、delta-yaw正确率和任务表现。使用`--no_wandb`时需用`tee`保存终端
  输出，否则没有可供后续分析的训练曲线。
- 服务器相机模式需要PhysX计算设备与Vulkan图形设备使用同一物理GPU；CUDA
  设备编号和Vulkan设备编号不保证一致。项目原先虽然解析
  `--graphics_device_id`，但没有传入环境，参数实际无效；现已接通并在
  `gym.create_sim()`前打印compute/graphics编号。多GPU相机模式应先用
  `vulkaninfo`确认Vulkan编号，再分别传`--device cuda:N`和
  `--graphics_device_id K`。随机箱相机地形的Delatin误差固定为`0.01 m`，
  防止继承原配置的`max_error_camera=2 m`后改变箱体几何。
- 多GPU相机训练不能在`CUDA_VISIBLE_DEVICES`只暴露一张卡后继续传物理
  `graphics_device_id>0`：Isaac Gym相机CUDA互操作会报
  `invalid device ordinal`。若CUDA与Vulkan编号一致，应暴露完整GPU列表，
  并让`--device cuda:N`、`--rl_device cuda:N`和
  `--graphics_device_id N`指向同一物理GPU。
- 192环境的随机箱视觉蒸馏进程在24GB GPU上实测占用约`16.5 GiB`（包含
  Isaac Gym非PyTorch显存）；如果同卡已有约`6.9 GiB`进程，会在首次深度卷积
  时OOM。该配置需要基本独占一张24GB卡；共享GPU时先确认进程归属，不能停止
  他人任务，必要时将相机环境数降至128。
- 相机深度后处理必须保持批量张量路径：Isaac Gym仍需逐传感器取得GPU tensor，
  但裁剪、高斯噪声、距离偏置、点丢失、随机遮挡、Bicubic缩放和历史buffer更新
  应在`torch.stack`后的全环境batch上执行，避免192环境逐张触发Python循环和
  GPU kernel。
- 推荐路径是先用最终 teacher policy 蒸馏得到深度相机 student，再导出匹配的
  base JIT 与 vision weights。teacher policy、视觉编码器与导出权重必须来自
  同一训练版本，不能把旧视觉权重和后续修改过的 Actor 混用。
- 本地单环境视觉回放支持显式传入 `--cpu_camera`：相机仍由NVIDIA/Vulkan
  渲染，但深度图通过Isaac Gym CPU readback读取，不经过
  `get_camera_image_gpu_tensor()`，可绕过本地PyTorch不支持RTX 5070 Ti
  `sm_120`的问题。该路径仅用于少量环境回放；服务器批量蒸馏继续使用默认
  GPU tensor相机路径。
- Viewer中的goal和脚部调试线会保留到下一次相机渲染，导致GUI相机回放的
  深度图出现球形waypoint轮廓；相机取图前现会先清理Viewer调试线，再在取图
  后重新绘制供人观察。无Viewer的`--headless`蒸馏从未绘制这些标记，因此
  已生成的student权重不受该问题污染。
- Viewer回放中按`R`会通过正常termination/reset链路重置当前观察环境，
  同时使视觉GRU收到done并清理episode、动作与深度历史；该快捷键不改变
  headless训练行为。
- 随机箱视觉任务已恢复原生`2.0 m`深度有效距离，并关闭`2.0～3.0 m`
  额外高斯噪声和无效点；两米以内既有相机噪声保持不变。通用远距离噪声
  实现仍保留但该任务不启用。深度GRU会按episode的`dones`逐环境清空，
  避免跨布局残留；`delta_yaw_ok`使用`abs(delta_yaw) < 0.6`，其余yaw
  蒸馏语义暂不改变。
- 独立泛化评估任务`go2_random_box_eval`使用固定未参与训练的seed
  `20260724`生成一条`30×4 m`十箱路线：横向偏移`±0.5 m`、箱间距
  `0.3～1.0 m`、地面粗糙度固定为`±0.03 m`。该任务默认仅1个环境和1条
  物理赛道，用于可复现地比较teacher/student，不修改`go2_random_box`、
  `go2_random_box_clean`或原生`go2`训练分布。评估路线的第2→3、第4→5
  和第8→9箱间距固定为`0.35 m`，第4箱高度固定为`0.40 m`。
  该任务另提供逻辑布局1，完整复用训练分布的原始布局130（6箱、强粗糙度档
  `0.02～0.06 m`）；逻辑布局2复用布局0的十箱序列，将每个箱体长宽各缩短
  `0.30 m`的初版已调整为：箱长仍精确缩短`0.30 m`，箱宽相较上一版逐箱
  增加`0.20 m`并限制在`0.90～1.40 m`，箱高限制在`0.20～0.50 m`，箱间净空限制在
  `0.20～0.70 m`。布局2将第3、6、10箱设为`0.45 m`，第5、9箱设为
  `0.20 m`，形成四组低→高；第1→2、第4→5、第6→7和第8→9箱四组
  高→低变化前均保留`0.70 m`净空。第3箱横向偏移从`+0.15 m`改为
  `-0.25 m`，即面对世界`+x`时向右移动`0.40 m`；第5箱横向偏移从
  `+0.45 m`改为`+0.25 m`，即向右移动`0.20 m`。使用
  `--random_box_layout 0/1/2`在三条评估路线间切换。
- 当前仓库有视觉蒸馏与 JIT 导出代码，但不包含完整、已验证的 Go2 真机低层
  控制程序；真机侧仍需适配 Unitree SDK2、D435i 深度输入、观测顺序、关节
  顺序/符号、动作缩放、默认姿态、PD 参数、控制频率、超时 watchdog 和急停。
- 上真机顺序必须是离线权重等价检查、只读 dry-run、悬挂/支架检查、平地低速
  测试、低矮单障碍测试，最后才是五箱；低层控制前必须避免与原生 sport mode
  同时发命令。
- `play_jit.py` 已改为从标准导出目录加载 `base_jit.pt` 与
  `vision_weight.pt`，默认单环境并保留随机箱任务地形；本地 Docker 使用
  `--device cpu --pipeline cpu --use_camera --cpu_camera` 可串联深度渲染、
  34维视觉输出、12维动作和 PhysX 回放。布局0已完成250步窗口与录制验证，
  生成的视频为1600×900、50 FPS、250帧，视觉和动作全程通过有限值检查。
