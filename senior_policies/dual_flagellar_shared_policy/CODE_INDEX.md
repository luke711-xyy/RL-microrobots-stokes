# `dual_flagellar_shared_policy` Code Index

## 1. 模块定位

这个目录实现双机器人高层编队训练的“真正参数共享多智能体版”：

- 物理层仍然是双机器人同一流体环境、联合求解
- 高层控制不再是单个 joint agent 输出 `Discrete(9)`
- 改为两个 agent：
  - `robot_1`
  - `robot_2`
- 两个 agent 各自输出自己的高层 primitive 动作 `Discrete(3)`
- 训练时两个 agent 共享同一套高层 policy 权重 `shared_policy`

一句话区分旧分支：

- `dual_flagella_self_propel`：双机器人、单 agent、联合动作
- `dual_flagellar_shared_policy`：双机器人、双 agent、共享 policy

## 2. 文件职责

| 文件 | 作用 |
| --- | --- |
| `discretization.py` | 生成求解器依赖的 `.pt` 预处理文件 |
| `calculate_v.py` | 双机器人联合流体求解，保留与单体维护分支一致的历史平面映射约定 |
| `swimmer.py` | 多智能体高层环境，负责底层 primitive 调用、team reward 计算、轨迹与诊断字段输出 |
| `train.py` | RLlib PPO 多智能体训练入口，配置 `shared_policy`，写 `TRAINING_PARAMS.md` 和 TensorBoard |
| `visualize_dual_flagella.py` | 可视化双 agent 共享策略；每个宏步分别为两个机器人取动作并逐子步播放 |
| `CODE_INDEX.md` | 当前目录索引 |
| `WORKLOG.md` | 当前目录工作日志 |

## 3. 运行链路

1. 运行 `discretization.py`
2. 生成 `calculate_v.py` 导入时必需的 `.pt` 文件
3. 运行 `train.py`
4. `train.py` 导入 `swimmer.py`
5. `swimmer.py` 在每个宏步内部调用 `calculate_v.py::RK_dual`

注意：

- `.pt` 文件是硬依赖，不是可选缓存
- 缺失时训练和可视化都会在导入阶段失败

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

每个 agent 观测维度固定为 `12`：

1. 自己的真实质心 `x, y`
2. 自己的平均朝向 `heading`
3. 自己当前 primitive 的 one-hot(3)
4. 对方相对自己的 `dx, dy`
5. 对方平均朝向 `heading`
6. 对方当前 primitive 的 one-hot(3)

### Reward 语义

当前版本采用“共享团队奖励”：

- 两个 agent 收到同一个 reward
- reward 公式完全沿用旧双体分支

## 5. 奖励构成

在每个宏步结束后计算：

```python
forward_reward = FORWARD_REWARD_COEF * avg_dx

err_x = abs(delta_x - FORMATION_TARGET_DX)
err_y = abs(delta_y - FORMATION_TARGET_DY)
shape_error = SHAPE_ERROR_X_WEIGHT * err_x + SHAPE_ERROR_Y_WEIGHT * err_y

trend_weight = trend_gate(shape_error)
anchor_weight = 0.5 + (SHAPE_ANCHOR_NEAR_MULTIPLIER - 0.5) * (1.0 - trend_weight)

shape_trend_reward = trend_weight * SHAPE_TREND_REWARD_COEF * (prev_shape_error - shape_error)
shape_anchor_penalty = -anchor_weight * SHAPE_ANCHOR_PENALTY_COEF * shape_error

team_reward = forward_reward + shape_trend_reward + shape_anchor_penalty
```

环境 `step()` 返回：

```python
rewards = {
    "robot_1": team_reward,
    "robot_2": team_reward,
}
```

## 6. 当前关键常量

| 常量 | 当前值 | 作用 |
| --- | --- | --- |
| `LOW_LEVEL_HOLD_STEPS` | `25` | 一个宏步包含的底层子步数 |
| `MACRO_HORIZON` | `50` | 每个 episode 的宏步数 |
| `ROBOT1_INIT` | `(4.0, -0.3)` | 机器人 1 初始位置 |
| `ROBOT2_INIT` | `(4.0, 0.3)` | 机器人 2 初始位置 |
| `FORMATION_TARGET_DX` | `0.0` | 目标相对 x 间距 |
| `FORMATION_TARGET_DY` | `2.0` | 目标相对 y 间距 |
| `FORWARD_REWARD_COEF` | `50.0` | 前进奖励系数 |
| `SHAPE_ERROR_X_WEIGHT` | `30.0` | x 偏差权重 |
| `SHAPE_ERROR_Y_WEIGHT` | `20.0` | y 偏差权重 |
| `SHAPE_TREND_REWARD_COEF` | `10.0` | 队形趋势奖励系数 |
| `SHAPE_ANCHOR_PENALTY_COEF` | `0.2` | 锚定惩罚系数 |
| `SHAPE_TREND_FADE_LOW` | `3.0` | 趋势奖励开始淡出阈值 |
| `SHAPE_TREND_FADE_HIGH` | `8.0` | 趋势奖励满权阈值 |
| `SHAPE_ANCHOR_NEAR_MULTIPLIER` | `2.0` | 靠近目标队形时的 anchor 增强系数 |

## 7. 训练配置

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
- `robot_1` 和 `robot_2` 都映射到同一个 `shared_policy`
- `count_steps_by = "env_steps"`

## 8. 可视化语义

`visualize_dual_flagella.py` 的核心逻辑：

- 恢复高层 checkpoint
- 取出 `shared_policy`
- 每个宏步分别对 `robot_1` / `robot_2` 的本地观测调用一次共享策略
- 组合成 `action_dict`
- 送入多智能体环境一步
- 按内部缓存的 `last_substep_frames` 逐子步播放

## 9. 重要诊断字段

训练回调、TensorBoard 和可视化都依赖这些环境字段：

- `last_forward_reward`
- `last_shape_trend_reward`
- `last_shape_anchor_penalty`
- `last_shape_error`
- `last_prev_shape_error`
- `last_trend_weight`
- `last_anchor_weight`
- `last_delta_x`
- `last_delta_y`
- `last_err_x`
- `last_err_y`
- `last_macro_action`
- `last_macro_action_names`

## 10. 快速定位

想改多智能体动作/观测接口：

- 看 `swimmer.py`

想改 shared policy 的 RLlib 配置：

- 看 `train.py::build_ppo_config`
- 看 `visualize_dual_flagella.py::build_config`

想改团队奖励：

- 看 `swimmer.py::step`

想改底层 primitive 调用：

- 看 `swimmer.py::_compute_low_level_action`

想改双体物理求解：

- 看 `calculate_v.py`

想查每次训练自动快照与参数记录：

- 看 `train.py::snapshot_current_visualizer`
- 看 `train.py::snapshot_current_swimmer`
- 看 `train.py::write_training_run_markdown`
