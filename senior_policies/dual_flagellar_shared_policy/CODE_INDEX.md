# `dual_flagellar_shared_policy` Code Index

## 1. 模块定位

这个目录实现双机器人高层“参数共享多智能体导航”训练：

- 物理层仍然是双机器人处于同一流体环境中的联合求解
- 高层控制是两个 agent：
  - `robot_1`
  - `robot_2`
- 两个 agent 各自独立决策高层 primitive
- 训练时共享同一套高层 policy 权重 `shared_policy`
- 当前任务模式固定为：共享目标点导航

## 2. 文件职责

| 文件 | 作用 |
| --- | --- |
| `discretization.py` | 生成求解器依赖的 `.pt` 预处理文件 |
| `calculate_v.py` | 双机器人联合流体求解，保持与单体维护分支一致的历史平面映射 |
| `swimmer.py` | 多智能体高层环境，负责目标点导航观测、独立导航 reward、底层 primitive 调用和轨迹记录 |
| `train.py` | RLlib PPO 训练入口，配置 `shared_policy`，写 `TRAINING_PARAMS.md` 和 TensorBoard |
| `visualize_dual_flagella.py` | 恢复高层 checkpoint 并显示双机器人目标点导航过程 |
| `CODE_INDEX.md` | 当前目录索引 |
| `WORKLOG.md` | 当前目录工作日志 |

## 3. 运行链路

1. 运行 `discretization.py`
2. 生成 `calculate_v.py` 导入时必需的 `.pt` 文件
3. 运行 `train.py`
4. `train.py` 导入 `swimmer.py`
5. `swimmer.py` 在每个宏步内部调用 `calculate_v.py::RK_dual`

## 4. 多智能体接口

### Agent IDs

- `robot_1`
- `robot_2`

### 动作空间

每个 agent 都是：

- `spaces.Discrete(3)`

动作编号：

- `0 -> forward`
- `1 -> cw`
- `2 -> ccw`

### 观测空间

每个 agent 的观测维度固定为 `8`：

```python
[
    centroid_x,
    centroid_y,
    heading,
    prev_action_one_hot[0],
    prev_action_one_hot[1],
    prev_action_one_hot[2],
    goal_rel_x,
    goal_rel_y,
]
```

语义固定为：

- `heading`：上一宏步最后一个底层子步结束时的平均几何朝向
- `prev_action_one_hot`：上一宏步实际执行的高层动作
- `goal_rel_x, goal_rel_y`：目标点相对当前质心的位置

当前默认目标：

- `GOAL_POINT = (8.0, 0.0)`
- `GOAL_RADIUS = 0.3`

## 5. heading 定义

当前目录明确沿用：

- `compute_average_heading(state_array)`

它表示整条机器人链条各段全局朝向的平均值。当前同时用于：

- 观测中的 `heading`
- 导航 reward 中的角度误差
- 可视化里的朝向显示

## 6. 奖励构成

当前 reward 是“每个机器人各自的导航 reward”，不再使用旧的编队趋势奖励。

对机器人 `i`，在每个宏步末计算：

```python
delta_pos_i = centroid_end_i - centroid_start_i
goal_vec_i = goal_point - centroid_end_i
goal_dist_i = ||goal_vec_i||
heading_i = compute_average_heading(state_i)
target_angle_i = atan2(goal_vec_i[1], goal_vec_i[0])
angle_error_i = wrap_to_pi(heading_i - target_angle_i)

progress_reward_i = NAV_PROGRESS_REWARD_COEF * dot(delta_pos_i, goal_unit_i)
angle_penalty_i = -NAV_ANGLE_PENALTY_COEF * (angle_error_i / pi) ** 2
reach_bonus_i = NAV_REACH_BONUS   # 仅首次进入目标半径时发一次
reward_i = progress_reward_i + angle_penalty_i + reach_bonus_i
```

特殊规则：

- 若本步结束时该机器人已在 `GOAL_RADIUS` 内：
  - `progress_reward_i = 0`
  - `angle_penalty_i = 0`
- `reach_bonus_i` 只在首次进入目标半径时发一次
- episode 只有在两个机器人都进入目标半径时才成功结束

## 7. 当前关键常量

| 常量 | 当前值 | 作用 |
| --- | --- | --- |
| `LOW_LEVEL_HOLD_STEPS` | `25` | 一个宏步包含的底层子步数 |
| `MACRO_HORIZON` | `50` | 每个 episode 的宏步数 |
| `ROBOT1_INIT` | `(4.0, -0.3)` | 机器人 1 初始位置 |
| `ROBOT2_INIT` | `(4.0, 0.3)` | 机器人 2 初始位置 |
| `GOAL_POINT` | `(8.0, 0.0)` | 共享目标点 |
| `GOAL_RADIUS` | `0.3` | 到达阈值 |
| `NAV_PROGRESS_REWARD_COEF` | `50.0` | 质心净位移投影奖励系数 |
| `NAV_ANGLE_PENALTY_COEF` | `10.0` | 角度平方惩罚系数 |
| `NAV_REACH_BONUS` | `20.0` | 首次进入目标半径奖励 |

## 8. 训练配置

当前高层 PPO 关键配置：

- `gamma = 0.995`
- `lr = 3e-4`
- `horizon = 50`
- `rollout_fragment_length = 50`
- `train_batch_size = 500`
- `sgd_minibatch_size = 100`
- `num_sgd_iter = 10`
- `entropy_coeff = 0.001`
- `use_lstm = False`

多智能体配置：

- 仅注册一个 policy：`shared_policy`
- `robot_1` 和 `robot_2` 都映射到 `shared_policy`
- `count_steps_by = "env_steps"`

## 9. 可视化语义

`visualize_dual_flagella.py` 的核心逻辑：

- 恢复高层 checkpoint
- 取出 `shared_policy`
- 分别对 `robot_1` / `robot_2` 的本地观测调用共享策略
- 组合成 `action_dict`
- 送入多智能体环境一步
- 按内部缓存的 `last_substep_frames` 逐子步播放
- 同时显示目标点、目标半径和两个机器人的独立导航指标

## 10. 重要诊断字段

训练回调、TensorBoard 和可视化依赖以下环境字段：

- `last_robot_rewards`
- `last_robot_progress_rewards`
- `last_robot_angle_penalties`
- `last_robot_angle_errors`
- `last_robot_goal_distances`
- `last_robot_headings`
- `last_robot_reached`
- `last_macro_action`
- `last_macro_action_names`
- `last_centroid1`
- `last_centroid2`

## 11. 快速定位

想改导航观测：

- 看 `swimmer.py::_get_single_obs`

想改每个机器人的导航 reward：

- 看 `swimmer.py::_compute_navigation_reward`
- 看 `swimmer.py::step`

想改 shared policy 的 RLlib 配置：

- 看 `train.py::build_ppo_config`
- 看 `visualize_dual_flagella.py::build_config`

想改目标点显示和可视化面板：

- 看 `visualize_dual_flagella.py::render_frame`

想查每次训练自动快照与参数记录：

- 看 `train.py::snapshot_current_visualizer`
- 看 `train.py::snapshot_current_swimmer`
- 看 `train.py::write_training_run_markdown`
