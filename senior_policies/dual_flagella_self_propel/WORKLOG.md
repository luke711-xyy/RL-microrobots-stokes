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
