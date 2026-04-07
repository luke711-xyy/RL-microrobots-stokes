# `dual_flagella_self_propel` Worklog

## Entry 001
- 时间：2026-04-06
- 本轮目标：建立双机器人高层编队训练分支
- 关键发现：
  - 现有仓库里没有可直接复用的成型双 flagella 联合求解分支
  - 旧高层目录里有“底层 checkpoint 驱动高层离散动作”的思路，但工程规范和训练记录方式落后
  - 当前维护的 `flagella_self_propel` 更适合作为新分支骨架
- 实际改动：
  - 新增 `senior_policies/dual_flagella_self_propel`
  - 新增双机器人联合求解版 `calculate_v.py`
  - 新增高层 joint env、训练入口、可视化入口
  - 新增 `CODE_INDEX.md` 与本工作日志
- 未决问题：
  - 需要实际结合底层 checkpoint 跑一次冒烟，确认 `Policy.from_checkpoint` 与当前 Ray 版本接口一致
  - 需要实际生成 `.pt` 文件后验证双体联合矩阵的数值稳定性
- 下一步：
  - 做运行时冒烟
  - 根据第一次训练日志调整 reward 权重或高层 PPO 参数

## Entry 002
- 时间：2026-04-07
- 本轮目标：排查双体高层训练在首个 `env.step()` 内出现的奇异矩阵报错
- 关键发现：
  - 报错发生在 `Calculate_velocity_dual()` 的 `torch.linalg.solve(M, R)`，说明退化的是约化刚体矩阵 `M`
  - `torch.linalg.solve(A.T, B_all.T)` 已经成功，因此首要问题不是 Stokeslet 大矩阵 `A`
  - 原来的 `build_dual_B_all()` 把双体平面列顺序拼成了 `[body1_x, body2_x, body1_y, body2_y]`
  - 但 `Q_total` 的行顺序是 `[body1_x, body1_y, body2_x, body2_y]`，两者基底不一致
- 实际改动：
  - 只修改 `calculate_v.py::build_dual_B_all()`
  - 将双体 B 的平面列顺序改成与 `Q_total` 一致的 `[body1_x, body1_y, body2_x, body2_y]`
  - 添加中文注释解释为什么这个顺序不能改乱
- 未决问题：
  - 尚未在带完整依赖和 `.pt` 预处理文件的目标环境中完成端到端复跑
  - 若修正后仍有病态矩阵，需要继续检查 `C1_total/C2_total` 的双体拼接语义
- 下一步：
  - 在目标机器重新运行 `train.py`
  - 若仍报线性代数错误，继续缩小到 `M` 的具体退化块

## Entry 003
- 时间：2026-04-07
- 本轮目标：处理双体求解器在修复矩阵基底后出现的状态维度广播错误
- 关键发现：
  - 新报错位置在 `Calculate_velocity_dual()` 的 `velon1[3:] = body1["action"]`
  - `body1["action"]` 是底层 primitive 的 9 维动作，但 `velon1` 被错误地初始化成了 `N+2=42` 维
  - 这里再次把求解器离散数 `N=40` 错当成了环境状态维度，和前面修过的 `swimmer.py` 属于同类问题
- 实际改动：
  - 将 `velon1` / `velon2` 的创建改为 `np.zeros_like(x1)` / `np.zeros_like(x2)`
  - 让双体求解器返回的状态增量维度始终与输入状态保持一致
- 未决问题：
  - 仍需在目标机器上继续复跑，确认下一层是否还有双体求解语义问题
- 下一步：
  - 重新运行 `train.py`
  - 若还有报错，继续沿着 `Calculate_velocity_dual()` 返回值与高层环境状态接口逐层核对
