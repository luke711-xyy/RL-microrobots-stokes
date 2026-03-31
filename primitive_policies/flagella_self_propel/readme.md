# Flagella Self-Propel — 多连杆鞭毛游泳器前进任务

## 项目概述

基于 **正则化 Stokeslet 方法 (Method of Regularized Stokeslets)** 的多连杆微型游泳机器人强化学习仿真。使用 **PPO (Ray RLlib + PyTorch)** 训练一个 **NL=10 连杆** 的鞭毛式游泳器在低雷诺数 (Stokes) 流体中实现自推进前进。

- **连杆数 (links)**: `NL = 10`
- **力点数 (sparse/force points)**: `N = NL * 4 = 40`，共 `N+1 = 41` 个点
- **速度点数 (dense/velocity points)**: `N_dense = NL * 8 = 80`，共 `N_dense+1 = 81` 个点
- **铰链自由度**: `NL - 1 = 9`（动作空间维度）
- **时间步长**: `DT = 0.01`，每个 RL step 内部做 `Ntime=10` 次 RK2 积分，总物理时间 `0.2`

---

## 文件结构总览

```
flagella_self_propel/
├── discretization.py   # 预处理：生成力点↔速度点映射，保存 .pt 文件（只需运行一次）
├── calculate_v.py      # 物理核心：Stokeslet 矩阵组装、速度求解、RK2 时间积分
├── swimmer.py          # Gym 环境：状态/动作空间、step/reset、奖励、轨迹记录
├── train.py            # 训练入口：PPO 超参配置 + 训练循环
├── readme.txt          # 原始简要说明（输出目录格式）
└── readme.md           # 本文件
```

运行顺序：先运行 `discretization.py` 生成 `.pt` 映射文件 → 再运行 `train.py` 启动训练。

---

## 1. discretization.py — 离散化预处理

**用途**：将游泳器的单位长度 `[0, 1]` 离散化为 sparse 力点和 dense 速度点，计算两套网格之间的最近邻映射关系，保存为 `.pt` 文件供 `calculate_v.py` 加载。

### 关键参数
| 参数 | 值 | 含义 |
|------|-----|------|
| `NL` | 10 | 连杆数 |
| `sparse_size` | `NL*4 = 40` | 力点数（Stokeslet 点） |
| `dense_size` | `NL*8 = 80` | 速度点数 |

### 核心逻辑 (L3-93)
1. 在 `[0,1]` 上均匀生成 `sparse_size+1` 个力点 `Xf` 和 `dense_size+1` 个速度点 `Xq` (L26-27)
2. 计算所有 dense→sparse 点的距离矩阵 `Distance` (L47)
3. 为每个 dense 点找最近的 sparse 点，生成映射标签 `Min_Distance_Label` (L48-51)
4. 统计每个 sparse 点对应多少个 dense 点 `Min_Distance_num` (L53-54)
5. 保存映射关系到 9 个 `.pt` 文件 (L83-92)

### 输出文件
| 文件名 | 内容 |
|--------|------|
| `Xf_match_q_fila.pt` | 每个 sparse 点对应的 dense 点 X 坐标 |
| `Yf_match_q_fila.pt` | 每个 sparse 点对应的 dense 点 Y 坐标 |
| `Zf_match_q_fila.pt` | 每个 sparse 点对应的 dense 点 Z 坐标 |
| `Xf_all_fila.pt` | 所有 sparse 点 X 坐标 |
| `Yf_all_fila.pt` | 所有 sparse 点 Y 坐标 |
| `Zf_all_fila.pt` | 所有 sparse 点 Z 坐标 |
| `Min_Distance_Label_Fila.pt` | dense→sparse 的 0/1 映射矩阵 (dense_size+1, sparse_size+1) |
| `Min_Distance_num_fila.pt` | 每个 sparse 点对应的 dense 点数量 |
| `Correponding_label_fila.pt` | 对应标签矩阵 |

---

## 2. calculate_v.py — 物理计算核心

**用途**：实现正则化 Stokeslet 方法求解多连杆游泳器在 Stokes 流中的速度，包括矩阵组装、线性系统求解和 Runge-Kutta 时间积分。

### 模块级初始化 (L1-98)
- 加载 `discretization.py` 生成的 9 个 `.pt` 映射文件 (L25-33)
- 预分配全局张量：Stokeslet 矩阵 `S_fila_fila`、Blakelet 矩阵 `B_fila_fila`、Pressurelet 矩阵 `P_fila_fila`、系统矩阵 `A_fila_fila`、压力矩阵 `PA_fila_fila` 等 (L47-91)

### 关键函数

#### Green 函数计算

| 函数 | 行号 | 功能 |
|------|------|------|
| `pressurelet_fila_fila(x,y,z,e)` | L101-111 | 计算正则化 Pressurelet（压力基本解），填充 `P_fila_fila` |
| `stokeslet_fila_fila(x,y,z,e)` | L118-142 | 计算正则化 Stokeslet（速度基本解），填充 `S_fila_fila` 的 3x3 张量分量 |
| `blakelet_fila_fila(x1,x2,x3,h,e)` | L149-168 | 计算 Blakelet（壁面镜像修正），填充 `B_fila_fila`（当前被注释未启用） |

> 参数 `e` 为正则化参数，`e = L * 0.1`，用于消除 Stokeslet 在原点的奇异性。

#### 矩阵组装

| 函数 | 行号 | 功能 |
|------|------|------|
| `M1M2(e)` | L177-361 | **核心组装函数**：计算所有点对间距 → 调用 Stokeslet/Pressurelet → 求和降维 → 组装系统矩阵 `A` 和压力矩阵 `PA`，返回 `A/(8πμ)` 和 `PA/(8πμ)` |

#### 运动学矩阵

| 函数 | 行号 | 功能 |
|------|------|------|
| `MatrixQp(L, theta)` | L377-402 | 构建 sparse 网格上的位置矩阵 Q（将刚体运动 + 铰链角 → 各点位置）|
| `MatrixQp_dense(L, theta)` | L404-426 | 构建 dense 网格上的位置矩阵 Q |
| `MatrixQ(L, theta, Qu, Q1, Ql, Q2)` | L366-374 | 将 Q 的子矩阵拼接为完整 Q 矩阵 |
| `MatrixB(L, theta, Y)` | L429-489 | 构建约束矩阵 B：力/力矩平衡（自由游动，合力合力矩为零）|
| `MatrixC(action_absolute)` | L491-499 | 构建控制矩阵 C：将 (Vx, Vy, ω, 铰链角速度) 映射到广义速度 |
| `MatrixD_sum(beta_ini, absU)` | L504-532 | 构建累积角度→位置的速度映射矩阵 D |
| `MatrixD_position(beta_ini, Xini, Yini, L)` | L535-547 | 构建从铰链角到各节点 XY 位置的映射 |

#### 速度求解

| 函数 | 行号 | 功能 |
|------|------|------|
| `Calculate_velocity(x, w, x_first)` | L551-660 | **核心求解函数**。流程：`initial()` 初始化 → 更新点位置到 Stokeslet 网格 → `MatrixB` + `M1M2` 组装 → `torch.linalg.solve` 求解 → 得到质心平动速度 `(Vx, Vy)` 和转动角速度 `ω` → 计算压力差 |
| `initial(x, w, x_first)` | L664-760 | 从状态向量 `x` 和动作 `w` 初始化所有几何/运动学量：连杆角度累加、theta 向量、Q 矩阵、绝对角速度 |
| `initial_dense(x, w, x_first)` | L765-803 | 同 `initial` 但用 dense 网格，仅返回位置 |

#### 时间积分

| 函数 | 行号 | 功能 |
|------|------|------|
| `RK(x, w, x_first)` | L809-852 | **Runge-Kutta 2 阶时间积分**。每个 RL step 调用一次，内部循环 `Ntime=10` 次，每次步长 `part_time=0.02`。返回：新状态 `xc`、X 位移 `Xn`、速度 `r`、首端位移 `x_first_delta`、各节点位置 `Xp/Yp`、压力差/压力值 |

### 速度求解数学流程
```
A · f = v_boundary        (Stokeslet 系统：已知边界速度求力)
B · f = 0                 (自由游动约束：合力/合力矩为零)
v_points = Q · (C1·U + C2) (运动学：刚体运动+铰链角速度→各点速度)

联立求解 → U = (Vx, Vy, ω)^T  (质心速度和角速度)
```

---

## 3. swimmer.py — Gym 环境

**用途**：定义 OpenAI Gym 兼容的强化学习环境，封装游泳器物理模型。

### 关键常量 (L21-33)

| 常量 | 值 | 含义 |
|------|-----|------|
| `N` | 10 | 连杆数（= NL） |
| `DT` | 0.01 | 时间步长 |
| `MAX_STEP` | 10000 | 最大步数 |
| `ACTION_LOW/HIGH` | -1 / 1 | 动作范围 |

### 类 `swimmer_gym(gym.Env)` (L70-418)

#### `__init__(self, env_config)` (L76-155)
- **动作空间**: `Box(-1, 1, shape=(N-1,))` = 9 维连续动作（各铰链角速度）
- **观测空间**: `Box(-10000, 10000, shape=(N-1,))` = 9 维（各铰链相对角度）
- **状态向量** `self.state`: shape `(N+2,)` = `[X质心, Y质心, θ首连杆全局角, β1, β2, ..., β9]`
- **初始位置**: `X_ini = -4, Y_ini = 4`
- **铰链角度范围**: `betamax = 2π/N`, `betamin = -betamax*0.5`
- `self.Xfirst`: 首连杆端点坐标 (2,)
- `self.XY_positions`: 所有节点位置 (N+1, 2)

#### `step(self, action)` (L167-372)
1. **动作预处理** (L190-219): 根据 `self.order` 决定动作方向；预测角度是否越界，越界则置零动作并给 `reward=-1`
2. **物理更新** (L234): 调用 `RK(self.state_n, w_tmp, self.Xfirst)` 进行时间积分
3. **奖励计算** (L257): `reward += pressure_diff * 10`（压力差驱动前进）
4. **轨迹记录** (L309-356): 累积到全局变量 `traj`/`traj2`/`trajp`，每 4000 步保存一次
5. **返回** (L370-372): `obs = state[3:]`（铰链角度），`reward`，`done`，`{}`

#### `reset(self)` (L376-404)
- 重置 `reward`、`done`、`order`
- **注意**：不重置 `self.state`（连续训练，状态延续）
- 返回 `self.state[3:]`

### 全局轨迹变量
| 变量 | 内容 |
|------|------|
| `traj` | 每行: `[X质心, Y质心, θ, β1...β9]` (N+2 列) |
| `traj2` | 每行: `[X质心, Y质心, X首端, Y首端]` (4 列) |
| `trajp` | 每行: 各点压力值 |

---

## 4. train.py — 训练入口

**用途**：配置 PPO 超参数并启动训练循环。

### PPO 关键超参数 (L41-88)

| 参数 | 值 | 说明 |
|------|-----|------|
| `gamma` | 0.9999 | 折扣因子（极高，关注长期回报） |
| `lr` | 0.0003 | 学习率 |
| `horizon` | 1000 | 每个 episode 最大步数 |
| `train_batch_size` | 1000 | 训练批大小 |
| `sgd_minibatch_size` | 64 | SGD 小批量 |
| `num_sgd_iter` | 30 | 每次训练的 SGD 迭代数 |
| `clip_param` | 0.1 | PPO 裁剪参数 |
| `use_lstm` | True | 使用 LSTM 网络 |
| `max_seq_len` | 20 | LSTM 序列长度 |
| `framework` | "torch" | 使用 PyTorch 后端 |

### 训练循环 (L147-167)
- 共 **2000** 轮训练迭代
- 每 **10** 轮保存一次 checkpoint 到 `policy/` 目录
- 创建 `traj/`、`traj2/`、`trajp/` 目录存储轨迹

---

## 数据流与调用关系

```
discretization.py (预处理，运行一次)
    └─ 生成 .pt 映射文件

train.py (训练入口)
    ├─ 导入 swimmer.py :: swimmer_gym
    ├─ 配置 PPO 并创建 trainer
    └─ trainer.train() 循环
         └─ swimmer_gym.step(action)
              └─ calculate_v.py :: RK(state, action, Xfirst)
                   └─ Calculate_velocity(x, w, x_first)  [×10 RK2 子步]
                        ├─ initial() → 几何/运动学初始化
                        ├─ M1M2(e) → Stokeslet 矩阵组装
                        │    ├─ stokeslet_fila_fila()
                        │    └─ pressurelet_fila_fila()
                        ├─ MatrixB() → 约束矩阵
                        ├─ MatrixQ() → 运动学映射
                        ├─ MatrixC() → 控制矩阵
                        └─ torch.linalg.solve() → 求解速度
```

---

## 输出目录说明

| 目录 | 内容 |
|------|------|
| `traj/` | `traj_0.pt, traj_1.pt, ...` — 每 4000 步保存一次，每行: `[X质心, Y质心, θ全局角, β1...β9]` |
| `traj2/` | `traj2_0.pt, traj2_1.pt, ...` — 每行: `[X质心, Y质心, X首端, Y首端]` |
| `trajp/` | `trajp_0.pt, trajp_1.pt, ...` — 每行: `[首端压力, 各铰链压力..., 末端压力]` |
| `policy/` | RLlib checkpoint，编号越大训练越充分 |

所有 `traj*/` 目录下的文件按文件名中的索引顺序拼接即为完整轨迹数据。
