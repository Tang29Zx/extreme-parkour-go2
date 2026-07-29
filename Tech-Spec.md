# Extreme Parkour Go2 五箱能力测试技术规格

## 方案

- 在目标仓库原生 `Terrain` heightfield 流程中增加 `five_box` 类型。
- 五箱生成逻辑放入独立模块，按 `seed=0` 和布局编号生成4套可复现布局。
- `Go2FiveBoxCfg` 继承 `Go2ParkourCfg`，只覆盖地形、随机化和 viewer；
  `Go2FiveBoxCfgPPO` 保持网络结构不变。
- waypoint 使用五个箱体中心和第五箱后平地，共6个顺序目标。
- `play.py` 对 `go2_five_box` 保留任务地形，不执行原有随机复杂地形覆盖。

## 几何参数

- 赛道：`18.0 × 2.0 m`；出生点：局部 `(0.6, 1.0, 0)`。
- 箱体：五个 `1.2 × 1.2 m`，当前能力诊断高度依次为
  `0.20/0.30/0.40/0.40/0.40 m`。第五箱由原始 `0.50 m` 暂降至
  `0.40 m`，用于隔离预训练策略的高度能力边界。
- 第一段助跑：`0.8～1.2 m`。
- 后续间距：dense `0.45～0.75`、normal `0.75～1.20`、sparse
  `1.20～1.60 m`，权重 `0.2/0.6/0.2`；高度不低于 `0.4 m`
  的箱体后间距至少 `0.8 m`。能力回放确认预训练策略会在第四、第五箱
  之间尝试直跳；当前将第五箱前间距单独设为 `1.60～2.00 m`，使第五箱
  在第四箱沿附近超出前向扫描范围，用于验证策略能否先下地再接近。

## 验证

- 单元测试验证布局重现、箱体边界、尺寸高度、间距与 waypoint。
- 运行 Python 语法检查和少量 Docker CPU 环境推理。
- 回归验证现有 `go2` 仍可注册和加载配置。

## 随机多箱扩展

- 新增独立 `random_box` terrain builder 和 `Go2RandomBoxCfg`，固定使用
  `25×4 m`、`10×40` 物理赛道与 7 个 goal 槽位。
- 400 条布局中 2～6 箱各 80 条；少箱布局用重复出口补齐 goal，不补碰撞体。
- 箱体横向偏移为 `-1.0～1.0 m`；箱间距在 `0.10～1.50 m` 内按
  `0.05 m` heightfield网格均匀采样。
- 随机箱赛道复用原仓库 `add_roughness()`：400条布局精确分为120条平地、
  200条 `0～0.02 m` 轻噪声和80条 `0.02～0.06 m` 强噪声。类别随固定
  seed确定性打散；五箱任务仍保持平整地面。
- 运行时噪声和动作延迟由新增配置开关控制，默认关闭，因此旧任务不改变；
  新任务开启观测噪声、每环境 episode 延迟、摩擦/质量/质心/电机随机化和
  小幅推力。
- 深度相机噪声仅在 `use_camera=True` 时生效；相机训练不得覆盖随机箱地形。
- `Go2RandomBoxCleanCfg` 继承完整随机箱配置，仅覆盖噪声与延迟开关；任务名
  `go2_random_box_clean`，用于从29500先适应随机箱几何，再进入完整Sim-to-Real
  随机化。

## 随机箱视觉遮挡修正

- `go2_random_box` 已恢复继承原生任务的 `2.0 m` 深度有效距离，避免改变
  所有深度像素的归一化尺度。
- 已关闭 `2.0～3.0 m` 额外高斯噪声和无效点；两米以内原有相机噪声保持
  不变。通用远距离噪声实现仍保留，但该任务不启用。
- 深度GRU在环境结束时按`dones`逐环境清空，避免上一局布局记忆污染新一局。
- `delta_yaw_ok`仅补充绝对值判断，其他Teacher/Student yaw逻辑保持不变。

## 导出模型 Docker 回放

- `play_jit.py` 通过 `--jit_model_dir` 读取标准导出目录；相对路径以
  `LEGGED_GYM_ROOT_DIR` 为基准，目录内必须包含 `base_jit.pt` 和
  `vision_weight.pt`，缺失时在创建仿真前失败并报告完整路径。
- 回放继续使用导出模型内的 estimator、history encoder 和 actor backbone；
  深度网络保持 58×87 输入、34维输出，其中前32维作为深度 latent，后2维
  乘1.5后写入本体观测 yaw 槽位。
- 回放默认 1 个环境、1×1 地形；命令行传入的 `--num_envs`、`--rows`、
  `--cols` 和 `--random_box_layout` 优先。随机箱任务不得执行旧的混合
  parkour 地形覆盖逻辑。
- RTX 5070 Ti 本地命令使用 `--device cpu --pipeline cpu --use_camera
  --cpu_camera`。Isaac Gym 仍通过 NVIDIA/Vulkan 渲染，相机图像经 CPU
  readback 进入 PyTorch，避免执行不支持 `sm_120` 的 CUDA kernel。
- `--replay_steps` 控制有限回放步数；未指定时保持原来的长回放。`--record`
  仅在有 viewer 时启用，通过 Isaac Gym viewer 截帧并编码到
  `--record_path` 指定的 MP4。

### 验证与回滚

- 运行 Python 语法检查、参数/路径聚焦测试和 `git diff --check`。
- Docker 中以单环境、1×1随机箱、CPU相机完成至少数步：
  深度编码输出有限、动作输出有限、仿真正常退出。
- 录制模式验证 MP4 文件非空且 `ffprobe` 可读取。
- 回滚时恢复 `play_jit.py`、`helpers.py` 以及本节文档；导出权重和日志不变。

## 十箱泛化评估任务

- `Go2RandomBoxEvalCfg`继承`Go2RandomBoxCfg`，保持观测、动作、奖励和
  网络配置不变；仅关闭每8秒一次的水平推力，避免外力干扰可复现的路线比较，
  其余观测、相机和物理随机化继续启用。
- 地形固定为`30×4 m`、默认`1×1`物理赛道、11个顺序goal。逻辑布局0用
  独立seed `20260724`生成10个箱体和1个出口goal；逻辑布局1通过
  `layout_presets`解析为训练参数下的源布局130；逻辑布局2解析到布局0的
  seed/index，将长度采样范围平移`-0.30 m`，宽度采样范围限制为
  `0.90～1.40 m`，高度限制为`0.20～0.50 m`，常规箱间净空限制为
  `0.20～0.70 m`；显式覆盖可扩展到`0.90 m`。
- 箱体范围保持高度`0.10～0.50 m`、长度和宽度`0.8～1.5 m`；横向偏移
  改为`-0.5～0.5 m`，箱间距改为`0.3～1.0 m`。
- 固定布局通过可选`gap_overrides`将零基索引`2/4/8`箱前间距设为
  `0.35 m`，通过`height_overrides`将零基索引`3`的高度设为`0.40 m`；
  未配置覆盖项的训练任务保持原采样逻辑和随机序列。
- 地面粗糙度固定为`0.03 m`档，避免布局选择时偶然生成平地。
- 相机模式固定使用单行单列地形；`play.py`与`play_jit.py`将该任务视为
  随机箱任务，不能被旧parkour演示地形覆盖。
- `BaseTask`订阅Viewer的`R`键并设置当前`lookat_id`对应的手动reset标记；
  `LeggedRobot.check_termination()`将该标记并入正常`reset_buf`，确保位置、
  episode buffer、动作历史、深度历史及策略RNN通过同一done语义完成重置。
- `resolve_random_box_layout()`同时提供给箱体生成和粗糙度档位选择，保证
  布局1的箱数、几何和粗糙度类别都来自源布局130；额外goal以出口重复填充。
- 布局2的长度与布局0使用大小相同的离散采样集合，因此同一随机样本对应
  精确缩短`0.30 m`；箱宽、高度和箱间净空重新采样到各自限制范围，横向
  中心默认保持逐箱一致。高度覆盖将第3、6、10箱设为`0.45 m`，第4箱
  保持`0.40 m`，第5、9箱设为`0.20 m`，形成四组明确的低→高变化。
- 布局2的高→低变化中，第1→2、第4→5、第6→7和第8→9箱的箱前净空
  分别固定为`0.40/0.90/0.90/0.40 m`；第2→3和第5→6箱固定为
  `0.35 m`，第3→4、第7→8和第9→10固定为`0.25 m`。
- 布局2将第3箱横向中心从`+0.15 m`改为`-0.25 m`；面对世界`+x`
  时这是向右移动`0.40 m`。第5箱横向中心从`+0.45 m`改为
  `+0.25 m`，即向右移动`0.20 m`。
- 布局3复用源布局0的固定随机序列，将箱长和箱宽采样范围收紧为
  `0.50～0.95 m`与`0.80～1.10 m`。十箱高度显式固定为
  `0.20/0.50/0.15/0.48/0.20/0.50/0.15/0.45/0.15/0.50 m`。
- 布局3的横向中心显式固定为
  `0.00/+0.55/-0.55/+0.65/-0.60/+0.60/-0.65/+0.55/-0.55/+0.60 m`；
  箱前净空依次为
  `0.20/0.85/0.20/0.90/0.25/0.95/0.20/0.85/0.20 m`。
  该布局单独将地面粗糙度固定为`0.06 m`档；布局0和布局2仍为`0.03 m`
  档，布局1保持源布局130的训练粗糙度。
- 聚焦测试验证固定seed、10个碰撞箱、边界、goal数量、出口余量、布局1与
  原130逐箱完全一致、布局2长度差值及宽度/高度/间距范围、布局3的完整
  高度/间距/横向折返序列及粗糙度，以及原训练配置未被继承类修改。

## Go2 原生混合评估任务

- `Go2MixedCfg` 继承 `Go2ParkourCfg`，只覆盖环境数、地形网格、地形权重
  和回放相机网格；PPO 配置保持原网络与 checkpoint 兼容。
- 地形使用 `1×15` 网格。前五类原生地形权重各为 `1`，`random_box`
  权重为 `10`，归一化后十五列依次映射到 terrain type
  `15/16/18/19/20`，其余十列均为`22`。
- 五条随机箱列直接复用 `build_random_box_terrain()`，布局编号依次为
  `0～4`。每条固定7箱，配置5个唯一布局；相邻箱间净空、长宽、高度和
  横向偏移分别采样于 `0.20～1.00 m`、`0.50～1.00 m`、`0.20～0.50 m`
  和 `-1.00～+1.00 m`，地面粗糙度固定为 `±0.02 m`。
- 逻辑布局2（第3条新增箱子赛道）通过`layout_presets`只覆盖第1～4箱
  横向中心为`-0.25/+0.25/-0.25/+0.25 m`；其余采样参数和随机序列不变。
- 逻辑布局5～9通过preset切换回`Go2RandomBoxCfg`训练参数，并依次映射
  到固定随机抽样的源布局`51/186/357/130/221`，仅将箱数范围覆盖为
  `(7,7)`。原布局已有的`2/2/6/6/6`个箱体保持几何与顺序不变，后续箱体
  沿同一seed随机序列按原训练尺寸、间距和横移分布继续生成；粗糙度档仍
  使用对应源布局的训练档位。
- `Terrain.curiculum()` 增加可选 `fixed_terrain_difficulty`。仅
  `go2_mixed` 设置为 `1.0`，现有任务缺少该字段时完全沿用原随机或课程难度。
- 原生五类地形均按 `num_goals=8` 生成；十条箱子列均使用7个箱体中心和1个
  出口goal。定制七箱最坏长度为`16.1 m`；补齐后的五条训练布局出口分别为
  `16.10/16.50/16.55/15.10/15.55 m`，均满足18米赛道的末端余量约束。
- `play.py` 将 `go2_mixed` 作为需保留任务地形的独立评估任务，固定使用
  15 个环境；`play_jit.py` 未显式传入覆盖参数时使用 `15` 个环境和 `1×15`
  地形，显式 `--num_envs/--rows/--cols` 仍优先。
- `Go2MixedCfg.env.draw_all_goals=True`使`_draw_goals()`遍历全部环境绘制
  goal球体；未设置该开关的其他任务仍只绘制`lookat_id`。当前/下一目标
  方向箭头继续只属于`lookat_id`。
- `check_termination()`在重置前保留`goal_success_buf`和有效episode掩码；
  `cur_goal_idx >= num_goals`表示完成最后一个waypoint。摔倒和自然超时是
  有效失败episode，手动reset从分母排除。
- `reset_idx()`按配置的物理列区间，将每个有效结束episode的`0/1`结果写入
  `extras["episode"]`：列`0～4`对应`native_1_5_jump_success_rate`，列
  `5～9`对应`custom_6_10_average_success_rate`，列`10～14`不纳入统计。
- PPO runner在每个训练日志周期合并episode结果并取均值，输出到终端和W&B；
  某组在当前周期没有结束episode时跳过该字段，避免空tensor产生`NaN`。
  回放脚本不输出累计或逐赛道成功率明细。
- 聚焦测试验证配置权重、固定难度、十五列 terrain type 顺序、固定七箱布局
  的几何边界、五条训练布局原有箱体前缀不变、补齐后均为7箱和每列8个goal，并运行语法检查与
  `git diff --check`。

## 可复盘训练日志

- `LocalMetricsWriter` 以实验目录为唯一写入边界：`metrics.csv` 使用
  `timestamp/iteration/total_timesteps/wall_time_s/metric/value` 长表格式，
  `train.log` 保存去除终端颜色的迭代摘要，TensorBoard event 写入同一目录，
  `checkpoints.csv` 记录 checkpoint 与 iteration 的对应关系。
- PPO 与视觉蒸馏共用 episode 聚合和本地写入路径。聚合时取所有
  `ep_infos` key 的并集，缺失或空 tensor 跳过，避免不同 terrain group 在同一
  iteration 没有样本时产生 `KeyError` 或 `NaN`。
- `check_termination()`分别保存路线完成、自然超时、摔倒和手动重置状态；
  PPO 原有的 `time_out_buf` 语义保持不变，路线完成仍视作无 terminal reward
  的结束。
- `reset_idx()`在课程更新和机器人状态重置前采集 episode 结果。手动重置不
  进入统计；成功、摔倒和自然超时均以实际结束 episode 为分母。
- 通用 episode 指标为 `success_rate/fall_rate/timeout_rate/goals_reached/
  goal_progress/episode_duration_s/distance_traveled_m`。terrain class 使用稳定
  名称映射写入 `terrain_<name>_success_rate`，未知类别回退到数值编号。
- `run_manifest.jsonl` 每次启动追加一条 JSON，包含 CLI、任务名、项目/实验名、
  完整 env/train 配置和 Git 状态；不读取环境变量、`.env` 或凭据。
- checkpoint 新增 `total_timesteps`、`total_time`、`training_mode` 与格式版本。
  加载旧 checkpoint 时，累计 timestep 回退为
  `iteration × num_steps_per_env × num_envs`，累计时间回退为0。
- 验证包括本地写入器单元测试、episode 结果聚合测试、checkpoint 元数据测试、
  Python 语法检查和相关既有测试；不启动大规模仿真训练。

## Onboard 完整边界单箱评测

- 新增独立入口`legged_gym/scripts/evaluate_onboard_single_box.py`。脚本先加载训练
  仓库任务与相机实现，再从`--onboard_root`末尾加入搜索路径，只导入
  `joint_mapping.py`、`policy_context.py`、`real_control_safety.py`和
  `unitree_boundary.py`；不得导入onboard仓库内较旧的`legged_gym`副本。
- 固定单箱几何复用`configure_fixed_single_box()`，通过新增的保留随机化选项只覆盖
  几何和episode长度，不关闭`go2_random_box`已有出生、观测、深度和物理随机化。
- PhysX的`FL/FR/RL/RR`状态先编码为Unitree `FR/FL/RR/RL` LowState等价数据，
  再由生产端`decode_low_state()`和`build_policy_proprio()`生成53维本体观测；
  LowCmd目标经`encode_low_cmd()`后反算为环境actor动作。
- 本体随机噪声按任务配置施加到角速度、roll/pitch、关节位置/速度与接触观测；深度
  图由Isaac相机渲染并经过现有批量高斯噪声、距离偏置、点丢失、随机遮挡、bicubic
  缩放与两米归一化路径。视觉GRU按10 Hz更新，动作与安全边界按50 Hz更新。
- temporal context由生产端`update_proprio_history()`维护；`last_action`保存actor
  原始输出，即使模型动作裁剪或LowCmd目标保护修改了实际执行动作也不回写历史。
- 每个trial维护独立的接管、prime、transition、上次目标、接触历史和计数器。策略
  接入后的结束优先级为：生产安全异常、任务成功、PhysX摔倒、15秒超时。
- 诊断先按生产公式重建目标步长、机械限位与PD力矩上下界，再调用生产
  `constrain_policy_target()`取得最终命令；只有请求落在联合区间之外且对应分量构成
  实际联合边界时，才把该周期归因给相应限位。多类边界可在同一周期同时命中。
- JSON schema v1保存权重哈希、Git commit、几何、随机化范围、总体成功率、各终止
  原因、全局/逐关节裁剪周期以及逐trial结果；已有输出文件拒绝覆盖。

验证：纯函数测试覆盖保留随机化、约束归因与聚合；Docker中使用CPU PhysX、CPU
pipeline和CPU相机回读先做少量环境冒烟，再分别运行`0.25 m`和`0.45 m`的20局评测。
运行后检查JSON可解析、trial数完整、深度/视觉/动作全部有限，并执行相关单元测试、
Python语法检查和`git diff --check`。
