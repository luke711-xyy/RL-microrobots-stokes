# `dual_flagella_self_propel` Worklog

## Entry 001
- 时间：2026-04-06
- 本轮目标：建立双机器人高层编队训练分支
- 关键发现：
  - 仓库里没有现成可直接复用的双 flagella 联合求解分支
  - 旧高层目录有“底层 checkpoint 驱动高层离散动作”的思路，但工程规范较旧
  - 当前维护中的 `flagella_self_propel` 更适合作为新分支骨架
- 实际改动：
  - 新增 `senior_policies/dual_flagella_self_propel`
  - 新增双机器人联合求解版 `calculate_v.py`
  - 新增高层 joint env、训练入口、可视化入口
  - 新增 `CODE_INDEX.md` 与工作日志
- 未决问题：
  - 需要结合真实底层 checkpoint 做训练冒烟
  - 需要验证双体联合矩阵的数值稳定性

## Entry 002
- 时间：2026-04-07
- 本轮目标：排查双体高层环境在首个 `env.step()` 中出现的奇异矩阵问题
- 关键发现：
  - 报错发生在 `torch.linalg.solve(M, R)`，退化的是约化刚体矩阵 `M`
  - `torch.linalg.solve(A.T, B_all.T)` 已成功，首要问题不在大矩阵 `A`
  - 当时的 `B_all` 列顺序与 `Q_total` 行顺序不一致，双体平面基底存在错位
- 实际改动：
  - 首轮修正 `calculate_v.py` 中双体平面基底顺序
- 未决问题：
  - 仍需继续核对 `A / B_all / Q_total` 是否真正共用同一套基底

## Entry 003
- 时间：2026-04-07
- 本轮目标：处理双体求解器中的状态维度混用问题
- 关键发现：
  - `Calculate_velocity_dual()` 一度把求解器离散数 `N=40` 误当成环境状态维度
  - 这导致底层 primitive 的 9 维动作无法写回状态增量
- 实际改动：
  - 将 `velon1` / `velon2` 改为 `np.zeros_like(x1)` / `np.zeros_like(x2)`
  - 保证双体求解器返回的状态维度始终与输入状态一致
- 未决问题：
  - 还需继续核对双体流体耦合矩阵的块顺序

## Entry 004
- 时间：2026-04-07
- 本轮目标：按最小改动原则修正双体流体矩阵的基底顺序，并增强训练诊断输出
- 关键发现：
  - 单体求解器的平面块顺序应保持为 `[x, z, y_unused]`
  - 双体版此前不仅 `B_all` 顺序有误，`A` 的块放置与 `Q_total` 的行顺序也没有完全对齐
  - 这类基底错位会让 `M = (A^{-T} B^T)^T Q C1` 在语义上失真
- 实际改动：
  - 修正 `build_joint_stokeslet_matrix()`，使其块顺序回到与单体版一致的 `[x, z, y_unused]`
  - 固定 `build_dual_B_all()` 为 `[body1_x, body2_x, body1_z, body2_z, body_y_unused]`
  - 新增 `build_dual_Q_total()`，显式把两个单体 `Q` 重排到与 `B_all` 一致的双体平面基底
  - 在 `swimmer.py` 中新增 `last_centroid1 / last_centroid2`
  - 将训练日志改为每个高层宏步都打印，并输出两个机器人实时质心坐标
- 未决问题：
  - 仍需在目标训练机上继续冒烟，确认修正后不再出现异常坐标跳变

## Entry 005
- 时间：2026-04-07
- 本轮目标：继续排查双体约化刚体矩阵 `M` 仍然奇异的问题
- 关键发现：
  - 双体版虽然已经修正了 `A / B_all / Q_total` 的部分基底顺序，但 `Q_total` 仍直接使用了 `MatrixQp` 的原始输出
  - 单体正式求解里参与 `MT = AB @ Q` 的不是原始 `MatrixQp`，而是再经过 `MatrixQ(...)` 重排后的 `Q`
- 实际改动：
  - 在 `Calculate_velocity_dual()` 中分别构造两个单体正式求解所用的 `Q_single_1 / Q_single_2`
  - 再将这两个重排后的 `Q` 送入 `build_dual_Q_total()`
- 未决问题：
  - 如果修正后仍报 `M` 奇异，则需要继续检查 `C1_total / C2_total` 与 6 个刚体自由度列顺序是否完全一致

## Entry 006
- 时间：2026-04-07
- 本轮目标：从第一性原理重新复核双体 `B_all` 的拼接逻辑
- 关键发现：
  - 旧的双体 `B_all` 把 `body1_x / body2_x` 放在前 3 行，把 `body1_z / body2_z` 放在后 3 行
  - 这等于把“刚体约束的行语义”和“流体分量的列语义”打乱了
  - 正确逻辑应是：每个机器人的 3 行刚体约束都要同时看到本体的 `x` 列和 `z` 列
  - 本地直接计算初始构型下的 `M` 后，旧实现 `rank(M)=3`，修正该拼接后 `rank(M)=6`
- 实际改动：
  - 重写 `calculate_v.py::build_dual_B_all()`
  - 现在双体 `B_all` 的 6 行按机器人分块，列按 `[body1_x, body2_x, body1_z, body2_z, body_y_unused]` 排列
- 未决问题：
  - 仍需在目标训练机上继续跑环境检查和真实训练，确认动态过程中不再出现奇异矩阵

## Entry 007
- 时间：2026-04-08
- 本轮目标：让双体高层可视化按每个底层子步播放，而不是只在宏步末刷新
- 关键发现：
  - 旧的 `visualize_dual_flagella.py` 每次只渲染一次 `env.step(action)` 的末状态
  - 但高层环境一个 `step()` 内部固定执行 `100` 个底层子步，所以画面天然卡顿
- 实际改动：
  - 在 `swimmer.py` 中新增 `last_substep_frames`，每个底层子步缓存一帧几何状态
  - 在 `visualize_dual_flagella.py` 中改成逐帧播放 `last_substep_frames`
  - 可视化启动时先绘制一帧初始状态
- 未决问题：
  - 如果 `--speed` 沿用旧值，逐子步播放后整体可视化时长会明显增加
- 下一步：
  - 在目标机器上实际回放，确认流畅度和总播放速度是否合适

## Entry 008
- 时间：2026-04-08
- 本轮目标：降低双体可视化在高层步切换时的停顿
- 关键发现：
  - 逐子步播放虽然让单个宏步内部变流畅了，但高层步之间仍会因为同步计算下一宏步而卡住
  - 在不引入后台线程的前提下，最小改动方案是预加载多个宏步到缓存队列
- 实际改动：
  - 在 `visualize_dual_flagella.py` 中加入 2 个高层步的预加载队列
  - 先同步算好 2 个宏步，再开始播放；队列耗尽后再补下一批
- 未决问题：
  - 这种方案会把停顿从“每个宏步一次”变成“大约每 2 个宏步一次”，不是完全无缝
  - 如果仍需进一步消除停顿，下一步应考虑后台 producer-consumer 预取
- 下一步：
  - 在目标机器上实测缓存队列效果
## Entry 009
- 时间：2026-04-11
- 本轮目标：将双体高层一个宏步对应的底层子步数从 100 改为 25，同时保持每个 episode 的总底层步数不变。
- 关键发现：该分支的时间尺度由 `LOW_LEVEL_HOLD_STEPS` 和 `MACRO_HORIZON` 两个常量统一控制，训练配置、可视化和参数记录都读取这两个常量。
- 涉及文件：`swimmer.py`、`CODE_INDEX.md`
- 实际改动：将 `LOW_LEVEL_HOLD_STEPS` 从 `100` 改为 `25`，将 `MACRO_HORIZON` 从 `50` 改为 `200`，保持每个 episode 总低层步数仍为 `5000`。
- 未决问题：更频繁的高层动作切换可能改变编队控制难度，需要继续观察训练曲线和可视化表现。
- 下一步：检查 `TRAINING_PARAMS.md` 和 TensorBoard 曲线是否反映新的宏步时间尺度。
## Entry 011
- 时间：2026-04-14
- 本轮目标：把双体高层环境从 reset-free 改成每回合重置到固定出发点。
- 关键发现：原来的 `reset()` 只清计数器，不重建机器人几何状态，导致回合之间会延续上一个 episode 的位置与姿态。
- 涉及文件：`swimmer.py`、`train.py`、`CODE_INDEX.md`
- 实际改动：`reset()` 现在会重新调用初始化几何逻辑，把两个机器人恢复到 `ROBOT1_INIT` 和 `ROBOT2_INIT`，同时清空轨迹与 reward 诊断字段；`TRAINING_PARAMS.md` 中的 `reset_behavior` 说明同步改为硬重置。
- 未决问题：硬重置后训练分布会变窄，需要继续观察 reward 曲线是否更稳定。
- 下一步：在新训练 run 的 `TRAINING_PARAMS.md` 中确认 reset 语义已经正确记录。
## Entry 012
- 时间：2026-04-14
- 本轮目标：把双体编队 reward 从“当前距离偏差直接惩罚”改成“误差改善趋势 + 锚定项 + 前进项”，并同步训练回调指标。
- 关键发现：
- 直接对 `|Δx-目标|` 和 `|Δy-目标|` 做惩罚，只能看到当前偏差大小，不容易判断策略是在修正编队还是继续恶化。
- 更适合当前高层控制语义的量，是“这一宏步之后的编队误差，相对上一宏步到底变好还是变坏”。
- 涉及文件：`swimmer.py`、`train.py`、`CODE_INDEX.md`
- 实际改动：
- `swimmer.py` 改为记录 `forward_reward / shape_trend_reward / shape_anchor_penalty / shape_error / prev_shape_error / err_x / err_y`
- 总 reward 现为 `forward + trend + anchor`
- 终端日志从旧的 `DxPen / DyPen` 改成 `Trend / Anchor / ShapeErr / PrevShapeErr`
- `train.py` 的 TensorBoard / custom_metrics 同步改为新字段命名
- `TRAINING_PARAMS.md` 记录的环境参数同步改为新的 reward 系数
- 未决问题：`SHAPE_TREND_REWARD_COEF=4.0` 与 `SHAPE_ANCHOR_PENALTY_COEF=0.5` 还需要结合实际训练曲线再看是否需要微调。
- 下一步：重新启动训练，观察 TensorBoard 中 `shape_error` 与 `shape_trend_reward` 是否出现更清晰的改善趋势。
## Entry 013
- 时间：2026-04-14
- 本轮目标：同步你实测后的双体 reward 系数，并把高层 episode 从 200 步缩到 20 步。
- 关键发现：
- 原来的 `MACRO_HORIZON=200` 会让后半段轨迹明显发散，reward 容易持续积累成巨大负值，不利于高层策略收敛。
- episode 变短后，采样参数如果还保持旧尺度，就会出现“环境已经按短局训练，更新却还按长局攒批次”的时间尺度不一致。
- 涉及文件：`swimmer.py`、`train.py`、`CODE_INDEX.md`
- 实际改动：
- reward 常量改为 `FORMATION_TARGET_DY=2.0`、`FORWARD_REWARD_COEF=50.0`、`SHAPE_ERROR_X_WEIGHT=30.0`、`SHAPE_ERROR_Y_WEIGHT=20.0`、`SHAPE_TREND_REWARD_COEF=10.0`、`SHAPE_ANCHOR_PENALTY_COEF=0.2`
- `MACRO_HORIZON` 从 `200` 改为 `20`，单回合总底层步数同步变为 `500`
- 训练采样参数同步改为 `horizon=20`、`rollout_fragment_length=20`、`train_batch_size=400`、`min_sample_timesteps_per_iteration=400`
- 未决问题：新 reward 系数明显更激进，需要重新观察 `episode_reward_mean` 和 `shape_error` 是否稳定下降。
- 下一步：跑一轮新训练，看 20 步 episode 下是否还会在后半段出现明显失控游动。
