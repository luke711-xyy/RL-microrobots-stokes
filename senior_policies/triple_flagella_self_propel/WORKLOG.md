# `triple_flagella_self_propel` Worklog

## Entry 001
- 时间：2026-04-12
- 本轮目标：基于 `codex/senior_policies` 双体工程，新建三体高层编队训练目录
- 关键发现：
  - 现有双体工程已经稳定了高层动作结构、奖励拆分方式和联合矩阵拼接语义
  - 三体版本最稳妥的落地方式不是重写框架，而是在双体骨架上做最小扩展
  - 为了避免从仓库根目录运行时找不到 `.pt` 离散化资源，资源加载应改成相对脚本目录
- 实际改动：
  - 新增 `senior_policies/triple_flagella_self_propel`
  - 新增三体联合求解版 `calculate_v.py`
  - 新增三体高层环境 `swimmer.py`
  - 新增三体训练入口 `train.py`
  - 新增三体可视化入口 `visualize_triple_flagella.py`
  - 新增 `CODE_INDEX.md` 与工作日志
- 当前约束：
  - 高层 observation 固定为 `22` 维
  - 高层宏动作固定为 `27` 个 primitive 三元组
  - reward 只约束相邻 pair：`12` 与 `23`
- 未决问题：
  - 需要用真实底层 checkpoint 跑一次完整训练/可视化冒烟
  - 需要确认三体默认初始构型下 `M` 在动态过程中持续保持满秩
