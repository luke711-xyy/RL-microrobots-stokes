# `dual_flagella_self_propel` Worklog

## Entry 001
- 时间：2026-04-06
- 本轮目标：建立双机器人高层编队训练分支
- 关键发现：
  - 现有仓库里没有可直接复用的成型双 flagella 联合求解分支
  - 老高层目录里有“底层 checkpoint 驱动高层离散动作”的思路，但工程规范和训练记录方式落后
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
