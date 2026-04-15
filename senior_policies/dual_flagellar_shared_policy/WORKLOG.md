# `dual_flagellar_shared_policy` Worklog

## Entry 001
- 时间：2026-04-15
- 本轮目标：从原双体单 agent 版本平行新建真正的参数共享多智能体高层训练分支
- 关键发现：
  - 原双体高层分支虽然物理上是双机器人耦合，但训练上仍是单 agent 联合动作控制
  - 要实现“每个机器人独立决策但共享经验”，关键是将高层环境接口切到 RLlib `MultiAgentEnv`
- 实际改动：
  - 新建 `senior_policies/dual_flagellar_shared_policy`
  - 实现两个 agent 的 shared-policy 训练框架
  - 保留双体联合流体求解与底层 primitive 调用方式
- 未决问题：
  - 后续需要根据训练表现继续迭代高层观测和 reward 设计

## Entry 002
- 时间：2026-04-15
- 本轮目标：将新分支的高层观测和奖励从“编队保持”重构为“共享目标点导航”
- 关键发现：
  - 当前代码中的 `heading` 已有稳定定义：`compute_average_heading(state)`，适合直接沿用到导航任务
  - 目标点导航第一版不需要再显式输入队友信息，可以先只保留“自身状态 + 相对目标向量”
- 涉及文件：
  - `swimmer.py`
  - `train.py`
  - `visualize_dual_flagella.py`
  - `CODE_INDEX.md`
  - `WORKLOG.md`
- 实际改动：
  - 将任务模式固定为共享目标点导航
  - 每个 agent 的观测改为 8 维：
    - 自己的质心坐标
    - 平均几何朝向 heading
    - 上一宏步动作 one-hot
    - 相对目标向量
  - 删除旧的编队趋势奖励
  - 新增每个机器人的独立导航 reward：
    - 位移投影奖励
    - 角度平方惩罚
    - 首次到达目标半径奖励
  - 成功条件改为“双机器人都进入目标半径”
  - TensorBoard、自定义日志和可视化面板全部改为导航语义
  - 更新 `CODE_INDEX.md` 说明当前任务结构和常量
- 未决问题：
  - 需要在训练环境中进一步验证当前奖励系数是否需要调优
  - 如目标点位置或到达半径不合适，后续需要再做实验校准
- 下一步：
  - 做静态检查
  - 运行一次最小训练/可视化冒烟
