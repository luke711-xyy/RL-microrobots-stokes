# `triple_flagella_self_propel` Code Index

## 1. 目录目的
这个目录实现三机器人高层平行编队训练。

核心思路：
- 三个 flagella 机器人处于同一平面、同一流体求解环境
- 高层只有一个 joint PPO agent
- 高层动作是 27 个宏动作之一，对应三个机器人各自从 `{forward, cw, ccw}` 中选一个底层 primitive
- 每个高层 step 固定执行 25 个底层子步，然后再重新筛选 primitive

## 2. 文件职责

| 文件 | 职责 |
| --- | --- |
| `discretization.py` | 生成与单机器人分支一致的 `.pt` 离散化文件 |
| `calculate_v.py` | 三机器人联合流体求解；保留单机器人矩阵构造思路，只把三条链条并入同一线性系统 |
| `swimmer.py` | 高层 Gym 环境；负责底层 checkpoint 加载、宏动作解码、25 子步滚动、编队 reward |
| `train.py` | 高层 PPO 训练入口；记录 `TRAINING_PARAMS.md` |
| `visualize_triple_flagella.py` | 三机器人高层策略可视化 |
| `CODE_INDEX.md` | 当前目录代码索引 |
| `WORKLOG.md` | 关键改动工作记录 |

## 3. 关键接口

### 高层动作空间
- `spaces.Discrete(27)`
- 动作表定义在 `swimmer.py::MACRO_ACTION_TABLE`
- 顺序为三个机器人 primitive 的笛卡尔积

### 高层观测空间
- 维度固定为 `22`
- 组成：
  - 机器人1：`[centroid_x, centroid_y, average_heading] + primitive_one_hot(3)`
  - 机器人2：`[centroid_x, centroid_y, average_heading] + primitive_one_hot(3)`
  - 机器人3：`[centroid_x, centroid_y, average_heading] + primitive_one_hot(3)`
  - 全局相对位形：`[dx12, dy12, dx23, dy23]`

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
| `ROBOT1_INIT` | `(-4, 1)` | 机器人 1 初始位置 |
| `ROBOT2_INIT` | `(-4, 0)` | 机器人 2 初始位置 |
| `ROBOT3_INIT` | `(-4, -1)` | 机器人 3 初始位置 |
| `FORMATION_TARGET_DX12` | `0.0` | 1-2 邻接 pair 的目标相对 x 间距 |
| `FORMATION_TARGET_DX23` | `0.0` | 2-3 邻接 pair 的目标相对 x 间距 |
| `FORMATION_TARGET_DY12` | `1.0` | 1-2 邻接 pair 的目标相对 y 间距 |
| `FORMATION_TARGET_DY23` | `1.0` | 2-3 邻接 pair 的目标相对 y 间距 |

## 5. Reward 公式

高层 reward 在每个宏步结束后计算：

```python
avg_dx = (dx_robot1 + dx_robot2 + dx_robot3) / 3
dx12 = centroid1_x - centroid2_x
dx23 = centroid2_x - centroid3_x
dy12 = centroid1_y - centroid2_y
dy23 = centroid2_y - centroid3_y

reward = (
    avg_dx
    - abs(dx12 - 0.0)
    - abs(dx23 - 0.0)
    - 2 * abs(dy12 - 1.0)
    - 2 * abs(dy23 - 1.0)
)
```

## 6. 物理求解链路

1. 高层环境在每个宏步内固定 primitive 三元组
2. 三个机器人各自调用对应底层 checkpoint，连续输出 25 个低层动作
3. 每个低层动作调用一次 `RK_triple(...)`
4. `RK_triple(...)` 内部再执行 10 次数值子步
5. `Calculate_velocity_triple(...)` 把三条链条的离散点并入同一个线性系统求解刚体速度和流体耦合

## 7. 运行注意事项

- 本目录要求先运行 `discretization.py`
- 底层 checkpoint 由三个机器人共享，但每个机器人对每个 primitive 各自维护自己的 recurrent state
- 当前高层环境默认采用 reset-free episode reset，只清空计数器，不恢复机器人几何状态
- 三体矩阵 `M` 如果不满秩会直接报错，不会用 `pinv` 或 `lstsq` 掩盖拼接错误
