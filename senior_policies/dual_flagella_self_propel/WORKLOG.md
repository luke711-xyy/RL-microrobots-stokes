# `dual_flagella_self_propel` Worklog

## Entry 001
- 时间：2026-04-06
- 本轮目标：建立双机器人高层编队训练分支
- 关键发现：
  - 仓库里没有现成可直接复用的双 flagella 联合求解分支
  - 旧高层目录里有“底层 checkpoint 驱动高层离散动作”的思路，但工程规范和训练记录方式较旧
  - 当前维护中的 `flagella_self_propel` 更适合作为新分支骨架
- 实际改动：
  - 新增 `senior_policies/dual_flagella_self_propel`
  - 新增双机器人联合求解版 `calculate_v.py`
  - 新增高层 joint env、训练入口、可视化入口
  - 新增 `CODE_INDEX.md` 与本工作日志
- 未决问题：
  - 需要结合真实底层 checkpoint 跑一轮冒烟，确认 `Policy.from_checkpoint` 与当前 Ray 版本兼容
  - 需要在生成 `.pt` 预处理文件后验证双体联合矩阵的数值稳定性
- 下一步：
  - 做运行时冒烟
  - 根据第一轮训练日志调整 reward 权重或高层 PPO 参数

## Entry 002
- 时间：2026-04-07
- 本轮目标：排查双体高层训练在首个 `env.step()` 中出现的奇异矩阵报错
- 关键发现：
  - 报错发生在 `Calculate_velocity_dual()` 的 `torch.linalg.solve(M, R)`，说明退化的是约化刚体矩阵 `M`
  - `torch.linalg.solve(A.T, B_all.T)` 已经成功，因此首要问题不在大矩阵 `A`
  - 当时的 `B_all` 列顺序与 `Q_total` 行顺序不一致，双体平面基底存在错位
- 实际改动：
  - 首轮修正 `calculate_v.py` 中双体平面基底顺序
- 未决问题：
  - 仍需继续核对 `A / B_all / Q_total` 三者是否完全共用同一套基底，而不是只修正一处
- 下一步：
  - 对照单体求解器做第一性原理复核
  - 继续缩小双体联合矩阵的错误来源

## Entry 003
- 时间：2026-04-07
- 本轮目标：处理双体求解器中的状态维度混用问题
- 关键发现：
  - `Calculate_velocity_dual()` 中曾将求解器离散数 `N=40` 误当成环境状态维度，导致 `body["action"]` 的 9 维动作无法写入输出状态
  - 这与之前 `swimmer.py` 中把求解器离散数误当作 primitive 关节维数属于同类错误
- 实际改动：
  - 将 `velon1` / `velon2` 改为 `np.zeros_like(x1)` / `np.zeros_like(x2)`
  - 保证双体求解器返回的状态增量始终与输入状态维度一致
- 未决问题：
  - 还需要确认双体流体耦合矩阵的块顺序完全正确
- 下一步：
  - 继续对照单体版检查 `A / B_all / Q_total` 的拼接语义

## Entry 004
- 时间：2026-04-07
- 本轮目标：按最小改动原则修正双体流体矩阵基底顺序，并增强训练诊断输出
- 关键发现：
  - 单体求解器的平面块顺序应保持为 `[x, z, y_unused]`
  - 双体版此前不仅 `B_all` 顺序有误，`A` 的块放置与 `Q_total` 的行顺序也没有完全对齐
  - 这类基底错位会让 `M = (A^{-T} B^T)^T Q C1` 在语义上失真，即使数值上暂时可解，也会表现为坐标爆炸
- 实际改动：
  - 修正 `calculate_v.py::build_joint_stokeslet_matrix()`，使其块顺序回到与单体版一致的 `[x, z, y_unused]`
  - 将 `build_dual_B_all()` 固定为 `[body1_x, body2_x, body1_z, body2_z, body_y_unused]`
  - 新增 `build_dual_Q_total()`，显式把两个单体 `Q` 重排到与 `B_all` 一致的双体平面基底
  - 在 `swimmer.py` 中新增 `last_centroid1 / last_centroid2`
  - 将训练日志改为每一个高层宏步都打印一次，并输出两个机器人实时质心坐标
- 未决问题：
  - 仍需在目标训练机上继续冒烟，确认修正后不再出现异常坐标跳变
  - 若仍有爆炸，需要继续检查 `C1_total / C2_total` 与刚体自由度顺序是否完全一致
- 下一步：
  - 重新运行双体高层训练
  - 对照每个宏步打印的 `R1 / R2 / dX / dY` 判断是否仍有系统性发散

## Entry 005
- 时间：2026-04-07
- 本轮目标：继续排查双体约化刚体矩阵 `M` 仍然奇异的问题
- 关键发现：
  - 双体版此前虽然修正了 `A / B_all / Q_total` 的基底顺序，但 `Q_total` 仍然直接使用了 `_initial_single()` 中 `MatrixQp` 的原始输出
  - 单体正式求解里，参与 `MT = AB @ Q` 的并不是原始 `MatrixQp` 结果，而是再经过 `MatrixQ(...)` 重排后的 `Q`
  - 这意味着双体版在进入约化系统前，`Q` 的列语义仍与单体原始物理推导不一致，足以继续让 `M` 秩亏
- 实际改动：
  - 在 `calculate_v.py::Calculate_velocity_dual()` 中，先分别构造两个单体正式求解所用的 `Q_single_1 / Q_single_2`
  - 再将这两个重排后的 `Q` 送入 `build_dual_Q_total()`
- 未决问题：
  - 如果修正后仍报 `M` 奇异，则需要继续检查 `C1_total / C2_total` 与 6 个刚体自由度列顺序是否完全一致
- 下一步：
  - 在目标训练机上重新运行高层训练
  - 若仍报奇异矩阵，继续输出 `rank(M)` 与奇异值定位退化来源

## Entry 006
- 时间：2026-04-07
- 本轮目标：从第一性原理重新复核双体 `B_all` 的拼接逻辑
- 关键发现：
  - 之前的双体 `B_all` 把 `body1_x / body2_x` 放在前 3 行、`body1_z / body2_z` 放在后 3 行
  - 这等于把“刚体约束的行语义”和“流体分量的列语义”交叉打乱了
  - 单体 `B` 的正确语义应是：每个机器人的 3 行刚体约束都要同时看到本体的 `x` 列和 `z` 列
  - 本地直接计算初始构型下的 `M` 后，旧实现的 `rank(M)=3`；修正该拼接后，`rank(M)=6`
- 实际改动：
  - 重写 `calculate_v.py::build_dual_B_all()`
  - 现在双体 `B_all` 的 6 行按机器人分块，列按 `[body1_x, body2_x, body1_z, body2_z, body_y_unused]` 排列
- 未决问题：
  - 还需要在目标训练机上继续跑环境检查和真实训练，确认动态过程中不再出现奇异矩阵
- 下一步：
  - 重新运行高层训练
  - 如果还有新报错，再继续沿 `M` 的奇异值和运行时坐标轨迹往下查
