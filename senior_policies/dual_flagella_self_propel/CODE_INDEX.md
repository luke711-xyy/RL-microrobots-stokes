# `dual_flagella_self_propel` Code Index

## 1. 目录目的
这个目录实现双机器人高层编队训练。

核心思路：
- 两个 flagella 机器人处于同一平面、同一流体求解环境
- 高层只有一个 joint PPO agent
- 高层动作是 9 个宏动作之一，对应两个机器人各自从 `{forward, cw, ccw}` 中选一个底层 primitive
- 每个高层 step 固定执行 25 个底层子步，然后再重新筛选 primitive

## 2. 文件职责

| 文件 | 职责 |
| --- | --- |
| `discretization.py` | 生成与单机器人分支一致的 `.pt` 离散化文件 |
| `calculate_v.py` | 双机器人联合流体求解；保留单机器人矩阵构造思路，只把两条链条并入同一线性系统 |
| `swimmer.py` | 高层 Gym 环境；负责底层 checkpoint 加载、宏动作解码、100 子步滚动、编队 reward |
| `train.py` | 高层 PPO 训练入口；记录 `TRAINING_PARAMS.md` |
| `visualize_dual_flagella.py` | 双机器人高层策略可视化 |
| `CODE_INDEX.md` | 当前目录代码索引 |
| `WORKLOG.md` | 关键改动工作记录 |

## 3. 关键接口

### 高层动作空间
- `spaces.Discrete(9)`
- 动作表定义在 `swimmer.py::MACRO_ACTION_TABLE`
- 顺序为两个机器人 primitive 的笛卡尔积：
  - `(forward, forward)`
  - `(forward, cw)`
  - `(forward, ccw)`
  - `(cw, forward)`
  - `(cw, cw)`
  - `(cw, ccw)`
  - `(ccw, forward)`
  - `(ccw, cw)`
  - `(ccw, ccw)`

### 高层观测空间
- 维度固定为 `14`
- 组成：
  - 机器人1：`[centroid_x, centroid_y, average_heading] + primitive_one_hot(3)`
  - 机器人2：`[centroid_x, centroid_y, average_heading] + primitive_one_hot(3)`
  - 全局相对位形：`[Δx, Δy]`

### 训练 CLI
- `--forward_ckpt`
- `--cw_ckpt`
- `--ccw_ckpt`
- `--num_cpus`
- `--num_threads`

### 可视化 CLI
- 同训练 CLI
- 额外支持 `--checkpoint`、`--steps`、`--speed`、`--view_range`

## 4. 关键常量

| 名称 | 值 | 作用 |
| --- | --- | --- |
| `LOW_LEVEL_HOLD_STEPS` | `25` | 每个宏动作持续的底层子步数 |
| `MACRO_HORIZON` | `200` | 每个 episode 的宏步数 |
| `ROBOT1_INIT` | `(-4, 2)` | 机器人 1 初始位置 |
| `ROBOT2_INIT` | `(-4, -2)` | 机器人 2 初始位置 |
| `FORMATION_TARGET_DX` | `0.0` | 编队目标相对 x 间距 |
| `FORMATION_TARGET_DY` | `4.0` | 编队目标相对 y 间距 |
| `FORWARD_REWARD_COEF` | `1.0` | 平均前进项权重 |
| `DELTA_X_PENALTY_COEF` | `1.0` | `Δx` 偏离罚项权重 |
| `DELTA_Y_PENALTY_COEF` | `2.0` | `Δy` 偏离罚项权重 |

## 5. Reward 公式

高层 reward 在每个宏步结束后计算：

```python
avg_dx = 0.5 * (dx_robot1 + dx_robot2)
delta_x = centroid1_x - centroid2_x
delta_y = centroid1_y - centroid2_y

reward = avg_dx - |delta_x - 0| - 2 * |delta_y - 4|
```

对应调试字段：
- `last_forward_reward`
- `last_dx_penalty`
- `last_dy_penalty`
- `last_delta_x`
- `last_delta_y`

## 6. 物理求解链路

1. 高层环境在每个宏步内固定 primitive 组合
2. 两个机器人各自调用对应底层 checkpoint，连续输出 25 个低层动作
3. 每个低层动作调用一次 `RK_dual(...)`
4. `RK_dual(...)` 内部再执行 10 次数值子步
5. `Calculate_velocity_dual(...)` 把两条链条的离散点并入同一个线性系统求解刚体速度和流体耦合

## 7. 运行注意事项

- 本目录仍然要求先运行 `discretization.py`
- 不修正现有单机器人分支的历史平面映射问题，保持一致
- 底层 checkpoint 由两个机器人共享，但每个机器人对每个 primitive 各自维护自己的 LSTM 隐状态
- 当前高层环境默认采用 reset-free episode reset，只清空计数器，不恢复机器人几何状态
## 8. Reset 行为更新

- 当前双体高层环境不再采用 reset-free。
- 每个 episode 开始时，两个机器人都会硬重置回固定起点 `ROBOT1_INIT=(-4, 0.2)` 和 `ROBOT2_INIT=(-4, -0.2)`。
- 底层 primitive 的 recurrent state、质心轨迹缓存和上一回合的 reward 诊断字段也会随之清空。

## 9. 当前 Reward 语义更新（2026-04-14）

当前双体高层 reward 已从“直接惩罚当前相对距离偏差”改为“三项分解”：

```python
err_x = abs(delta_x - FORMATION_TARGET_DX)
err_y = abs(delta_y - FORMATION_TARGET_DY)
shape_error = SHAPE_ERROR_X_WEIGHT * err_x + SHAPE_ERROR_Y_WEIGHT * err_y

forward_reward = FORWARD_REWARD_COEF * avg_dx
shape_trend_reward = SHAPE_TREND_REWARD_COEF * (prev_shape_error - shape_error)
shape_anchor_penalty = -SHAPE_ANCHOR_PENALTY_COEF * shape_error

reward = forward_reward + shape_trend_reward + shape_anchor_penalty
```

当前环境里需要重点看的诊断字段：

- `last_forward_reward`：两个机器人当前宏步平均 x 前进收益
- `last_shape_trend_reward`：相对上一个宏步，编队误差是否在改善
- `last_shape_anchor_penalty`：当前编队误差本身的锚定惩罚
- `last_shape_error`：当前加权编队误差
- `last_prev_shape_error`：上一宏步的加权编队误差
- `last_err_x`：当前 `Δx` 相对目标的绝对误差
- `last_err_y`：当前 `Δy` 相对目标的绝对误差

训练回调和 TensorBoard 现在也按这组字段写入，不再使用旧的 `dx_penalty / dy_penalty` 命名。

## 10. 当前训练尺度更新（2026-04-14）

当前双体高层分支已经切到更短的 episode 尺度：

- `LOW_LEVEL_HOLD_STEPS = 25`
- `MACRO_HORIZON = 20`

这意味着：

- 每个高层 episode 只包含 `20` 个宏步
- 每个宏步仍固定执行 `25` 个底层子步
- 所以每个 episode 的总底层步数变为 `20 * 25 = 500`

为匹配这个更短的时间尺度，训练采样参数同步调整为：

- `horizon = 20`
- `rollout_fragment_length = 20`
- `train_batch_size = 400`
- `min_sample_timesteps_per_iteration = 400`

当前 reward 常量为：

- `FORMATION_TARGET_DX = 0.0`
- `FORMATION_TARGET_DY = 2.0`
- `FORWARD_REWARD_COEF = 50.0`
- `SHAPE_ERROR_X_WEIGHT = 30.0`
- `SHAPE_ERROR_Y_WEIGHT = 20.0`
- `SHAPE_TREND_REWARD_COEF = 10.0`
- `SHAPE_ANCHOR_PENALTY_COEF = 0.2`

## 11. 可视化连续播放开关（2026-04-14）

双体可视化脚本支持：

- `--reset_free_playback`

其语义是：

- 只在可视化端做“连续播放”
- 当环境命中 `done=True` 的 episode 边界时，不调用 `env.reset()`
- 只清空 `ep_step / done / reward` 等回合计数器，让机器人沿当前几何状态继续往前滚动

注意这不会改变训练语义：

- 训练仍然是硬重置 episode
- `--reset_free_playback` 只用于观察短局训练出来的策略，在回合边界之外是否还能继续稳定推进和维持编队
