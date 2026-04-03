# `flagella_self_propel` Code Index

## 1. Scope and purpose
This document is the working index for `primitive_policies/flagella_self_propel`, the primitive-policy branch used to learn forward self-propulsion for a multi-link flagellar swimmer.

Use this file first before reading source. It is optimized for:

- fast code navigation
- remembering file responsibilities
- locating reward, PPO, action/state, and physics-solver logic
- reducing repeated full-file reads in later tasks

This index only covers `flagella_self_propel`. It intentionally ignores the other primitive-policy branches for now.

## 2. Directory contents

| File | Role | Main outputs / side effects |
| --- | --- | --- |
| `discretization.py` | One-time preprocessing script that builds sparse-force to dense-velocity point mappings. | Writes 9 `.pt` files required by `calculate_v.py`. |
| `calculate_v.py` | Low-level hydrodynamics and time integration core. | Loads `.pt` mapping files on import; computes velocities, pressure, RK2 updates. |
| `swimmer.py` | Gym environment wrapper around the swimmer physics. | Defines reward, state/action spaces, trajectory buffers, and periodic `.pt` trajectory dumps. |
| `train.py` | PPO training entrypoint using Ray RLlib + PyTorch. | Creates `policy_<timestamp>/`, `traj/`, `traj2/`, `trajp/`; trains and checkpoints policy. |
| `visualize_self_propel.py` | Real-time policy visualizer. | Loads a checkpoint, runs the trained PPO policy, and renders the swimmer body, heading cues, and centroid trace. |
| `readme.txt` | Original short English note about generated output folders. | Reference only. |
| `readme.md` | Existing local overview for this branch. | Useful as prior context, but this file is the maintained index going forward. |

## 3. Runtime pipeline

The actual dependency order is:

1. Run `discretization.py`.
2. It writes the `.pt` mapping files into the current working directory.
3. `train.py` imports `swimmer.py`.
4. `swimmer.py` imports `RK` from `calculate_v.py`.
5. `calculate_v.py` immediately loads the `.pt` files at import time.
6. PPO training calls `swimmer_gym.step()`.
7. `step()` calls `RK(...)`.
8. `RK(...)` repeatedly calls `Calculate_velocity(...)`.
9. `Calculate_velocity(...)` assembles the linear system, solves for velocity/force, and returns pressure-related terms used by the reward.

Important: the `.pt` files are not optional cached artifacts. They are import-time prerequisites for `calculate_v.py`. If they do not exist in the working directory, importing the environment will fail before training starts.

## 4. Key parameters

### Geometry and discretization

| Symbol / variable | Value | Where |
| --- | --- | --- |
| `NL` | `10` links | [`calculate_v.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/calculate_v.py):40, [`discretization.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/discretization.py):4 |
| `N` | `NL * 4 = 40` sparse / force-point segments | [`calculate_v.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/calculate_v.py):41 |
| `N_dense` | `NL * 8 = 80` dense / velocity-point segments | [`calculate_v.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/calculate_v.py):40-41 |
| sparse points | `sparse_size + 1 = 41` | [`discretization.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/discretization.py):5-9 |
| dense points | `dense_size + 1 = 81` | [`discretization.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/discretization.py):4, [`discretization.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/discretization.py):11-13 |

### Environment and simulation

| Item | Value | Where |
| --- | --- | --- |
| action dimension | `N-1 = 9` | [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):86 |
| observation dimension | `N-1 = 9` | [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):87 |
| action range | `[-1, 1]` | [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):32-34 |
| `DT` | `0.01` | [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):30 |
| max step counter | `10000` | [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):29 |
| initial centroid | `(-4, 4)` | [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):102-103 |
| hinge upper bound | `2*pi/N` | [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):82 |
| hinge lower bound | `-0.5 * betamax` | [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):83 |

### RK integration

| Item | Value | Where |
| --- | --- | --- |
| `Ntime` | `10` substeps per RL step | [`calculate_v.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/calculate_v.py):818 |
| `whole_time` | `0.2` | [`calculate_v.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/calculate_v.py):819 |
| `part_time` | `whole_time / Ntime = 0.02` | [`calculate_v.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/calculate_v.py):820 |
| regularization `e` | `L * 0.1` | [`calculate_v.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/calculate_v.py):681, [`calculate_v.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/calculate_v.py):768 |

### PPO training

| Item | Value | Where |
| --- | --- | --- |
| Ray CPUs | CLI arg `--num_cpus`, default `5` | [`train.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/train.py):7, [`train.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/train.py):31 |
| PyTorch threads | CLI arg `--num_threads`, default `5` via env var | [`train.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/train.py):8-10 |
| `gamma` | `0.9999` | [`train.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/train.py):53 |
| `lr` | `0.0003` | [`train.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/train.py):54 |
| `horizon` | `1000` | [`train.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/train.py):55 |
| `train_batch_size` | `1000` | [`train.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/train.py):64 |
| `sgd_minibatch_size` | `64` | [`train.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/train.py):63 |
| `num_sgd_iter` | `30` | [`train.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/train.py):65 |
| `clip_param` | `0.1` | [`train.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/train.py):70 |
| `use_lstm` | `True` | [`train.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/train.py):89 |
| `max_seq_len` | `20` | [`train.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/train.py):90 |
| training iterations | `2000` | [`train.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/train.py):153 |
| checkpoint cadence | every 10 iterations | [`train.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/train.py):156-169 |

## 5. File-by-file index

### `discretization.py`

Purpose:

- builds sparse-force and dense-velocity grids on the unit-length swimmer centerline
- computes nearest-neighbor assignments from dense points to sparse points
- saves tensors later consumed by `calculate_v.py`

Important details:

- `NL=10`, `dense_size=80`, `sparse_size=40` at [`discretization.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/discretization.py):4-6
- force points are on `Xf`, velocity points are on `Xq`; both sit at constant `z=0.01`
- `N=3` at [`discretization.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/discretization.py):3 is not the swimmer link count; here it acts as an allocation width for local correspondence slots

Generated files:

- `Xf_match_q_fila.pt`
- `Yf_match_q_fila.pt`
- `Zf_match_q_fila.pt`
- `Min_Distance_Label_Fila.pt`
- `Xf_all_fila.pt`
- `Yf_all_fila.pt`
- `Zf_all_fila.pt`
- `Min_Distance_num_fila.pt`
- `Correponding_label_fila.pt`

Save locations:

- [`discretization.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/discretization.py):83-92

### `calculate_v.py`

Purpose:

- loads the discretization artifacts at import time
- constructs regularized Stokeslet / pressurelet matrices
- maps hinge actions into absolute link motion
- solves the mobility / force system with `torch.linalg.solve`
- performs RK2 time stepping

Import-time dependency:

- all nine `.pt` mapping files are loaded at [`calculate_v.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/calculate_v.py):25-33
- PyTorch thread count is set from `STOKES_NUM_THREADS` at [`calculate_v.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/calculate_v.py):42

Core solve path:

- `initial(...)` converts environment state + hinge rates into geometry and absolute-angle representations
- `M1M2(e)` assembles the linear system matrices
- `Calculate_velocity(...)` solves for force and rigid-body motion using `torch.linalg.solve` at [`calculate_v.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/calculate_v.py):605 and [`calculate_v.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/calculate_v.py):637
- `RK(...)` performs the outer RK2 integration loop and returns the updated state and pressure arrays

Returned physics terms used by the environment:

- `pressure_diff = pressure_end - pressure_start`
- `pressure_all` is the full pressure profile over swimmer points
- `x_first_delta` updates the tracked back-end endpoint `self.Xfirst`

### `swimmer.py`

Purpose:

- exposes the swimmer as a Gym-compatible environment
- defines the reward
- stores trajectory snapshots
- periodically flushes trajectories to disk

State layout:

- `self.state` has shape `(N+2,)`
- semantic layout: `[centroid_x, centroid_y, first_link_global_angle, hinge_angles...]`
- returned observation is `self.state[3:]`, not the full state

Action semantics:

- incoming action has 9 continuous components
- `self.order` can flip the sign convention for actions and returned observations, although the current branch initializes and resets `self.order` to `0`
- angle-limit checks can zero out the action and apply immediate negative reward if a predicted angle exceeds the allowed range

Reward terms:

- pressure reward: `pressure_diff.item() * 12` at [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):269-270
- direction penalty: compares two equal 30-step centroid-direction windows, `t-60 -> t-30` versus `t-30 -> t`, and gates the penalty by the recent 30-step net displacement instead of pressure magnitude at [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):272-295
- current displacement gate: `min(recent_norm / self.displacement_gate_ref, 1.0)` with `self.displacement_gate_ref = 0.05` at [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):163-164 and [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):294
- every 200 steps, debug logging now prints both `Disp30` and `Gate30` so the displacement threshold can be tuned from training logs
- total reward accumulated into `self.reward` at [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):305

Trajectory dumps:

- every 4000 env steps:
  - `traj/traj_<idx>.pt`
  - `traj2/traj2_<idx>.pt`
  - `trajp/trajp_<idx>.pt`
- flush logic at [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):391-397

### `train.py`

Purpose:

- CLI-driven PPO training entrypoint
- configures Ray CPU count and PyTorch thread count
- defines PPO hyperparameters
- creates output directories
- checkpoints policy periodically

Output folder behavior:

- policy root is timestamped as `policy_<YYYYMMDD_HHMMSS>` at [`train.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/train.py):25-26
- trajectory directories `traj/`, `traj2/`, `trajp/` are created if missing at [`train.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/train.py):137-148

### `visualize_self_propel.py`

Purpose:

- loads the latest or user-specified PPO checkpoint for this branch
- runs the trained policy in the self-propel environment with `explore=False`
- renders the swimmer body, centroid trace, centroid marker, and heading cues in a simple real-time window

Important details:

- parses `--num_cpus` and `--num_threads` before importing the environment so visualization matches the same thread-control pattern as training
- calls `os.chdir(BASE_DIR)` before importing `swimmer.py`, because the environment depends on the same working-directory assumptions as training
- auto-detects the latest `policy_*` checkpoint if `--checkpoint` is not provided
- accepts both older file-style checkpoint paths and newer RLlib directory-style checkpoints such as `.../10/checkpoint_000011`
- prints the current reward decomposition in both the terminal and the plot overlay
- stays close to the older reorient visualizer style instead of adding extra fluid-field rendering

## 6. Function and class index

### `calculate_v.py`

| Symbol | Line | Purpose | Called by |
| --- | --- | --- | --- |
| `pressurelet_fila_fila(x, y, z, e)` | 101 | Fills regularized pressurelet tensor for filament-force interactions. | `M1M2` |
| `stokeslet_fila_fila(x, y, z, e)` | 118 | Fills regularized Stokeslet tensor for velocity influence calculation. | `M1M2` |
| `blakelet_fila_fila(x1, x2, x3, h, e)` | 149 | Computes Blakelet wall-correction tensor; currently part of matrix machinery but not the branch's main active reward logic. | `M1M2` |
| `M1M2(e)` | 177 | Main matrix assembly routine for hydrodynamic operators and pressure operator. | `Calculate_velocity` |
| `MatrixQ(L, theta, Qu, Q1, Ql, Q2)` | 366 | Combines sub-blocks into the full kinematic mapping matrix. | `MatrixQp` |
| `MatrixQp(L, theta)` | 377 | Builds sparse-grid kinematic mapping from swimmer generalized coordinates to point positions. | `initial` |
| `MatrixQp_dense(L, theta)` | 404 | Dense-grid version of the kinematic mapping. | `initial_dense` |
| `MatrixB(L, theta, Y)` | 429 | Builds free-swimming force/torque balance constraints. | `Calculate_velocity` |
| `MatrixC(action_absolute)` | 491 | Maps rigid-body and hinge terms into generalized velocity coordinates. | `Calculate_velocity` |
| `MatrixD_sum(beta_ini, absU)` | 504 | Builds cumulative angle-to-velocity mapping terms. | `Calculate_velocity` |
| `MatrixD_position(beta_ini, Xini, Yini, L)` | 535 | Recovers point positions from cumulative angles. | `Calculate_velocity` |
| `Calculate_velocity(x, w, x_first)` | 551 | End-to-end velocity / force / pressure solve for one state-action pair. | `RK` |
| `initial(x, w, x_first)` | 665 | Converts env state and hinge actions into geometry and sparse-grid structures. | `Calculate_velocity` |
| `initial_dense(x, w, x_first)` | 765 | Dense-grid geometry initialization. | currently only local physics support |
| `RK(x, w, x_first)` | 809 | RK2 outer integrator for one RL step; returns new state, displacement, pressures, and point positions. | `swimmer_gym.step` |

### `swimmer.py`

| Symbol | Line | Purpose |
| --- | --- | --- |
| `swimmer_gym` | 71 | Main Gym environment class for PPO training. |
| `swimmer_gym.__init__(self, env_config)` | 77 | Initializes action/observation spaces, initial state, reward bookkeeping, trajectory buffers, and debug counters. |
| `swimmer_gym.seed(self, seed=None)` | 175 | Standard Gym seeding hook. |
| `swimmer_gym.step(self, action)` | 179 | Clips and validates actions, calls `RK`, computes reward, appends trajectory rows, and conditionally writes files. |
| `swimmer_gym.reset(self)` | 421 | Reset-free episode reset: clears reward/debug counters but does not restore the geometric state to the original initial condition. |
| `swimmer_gym._get_obs(self)` | 451 | Legacy helper returning a concatenation involving `self.reach_targets`; currently stale for this branch. |
| `swimmer_gym.render(self)` | 454 | Stub. |
| `swimmer_gym.close(self)` | 458 | Closes `viewer` if one exists. |

## 7. Output data index

### Policy checkpoints

- root: `policy_<timestamp>/`
- nested directories are created by RLlib `trainer.save(...)`
- later checkpoints with larger iteration numbers represent later training states

### `traj/`

Each row:

- centroid `X`
- centroid `Y`
- first-link global angle
- all hinge angles

Source:

- assembled in [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):345-359

### `traj2/`

Each row:

- centroid `X`
- centroid `Y`
- back-end point `X` (`self.Xfirst[0]`)
- back-end point `Y` (`self.Xfirst[1]`)

Source:

- assembled in [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):345-364

### `trajp/`

Each row:

- pressure at the first-link end
- pressures at hinge-related points
- pressure at the last-link end

Source:

- assembled in [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):376-379

## 8. Quick lookup table

| If you want to... | Go to |
| --- | --- |
| change PPO hyperparameters | [`train.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/train.py):49-94 |
| change Ray CPU count or PyTorch thread count | [`train.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/train.py):5-10, [`train.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/train.py):31, [`calculate_v.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/calculate_v.py):42 |
| change action / observation dimensions | [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):22, [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):86-87 |
| change hinge angle limits | [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):82-83, [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):212-226 |
| change initial swimmer position | [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):102-112 |
| change reward terms | [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):269-295 |
| remove or alter direction-stability penalty | [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):272-295 |
| inspect import-time file dependencies | [`calculate_v.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/calculate_v.py):25-33 |
| inspect the main hydrodynamic solve | [`calculate_v.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/calculate_v.py):551-660 |
| inspect RK integration | [`calculate_v.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/calculate_v.py):809-852 |
| inspect trajectory file writing | [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):391-397 |
| inspect checkpoint saving | [`train.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/train.py):153-169 |
| run or modify policy visualization | `visualize_self_propel.py` |
| adjust visualization window, playback speed, or trace length | `visualize_self_propel.py` via `--view_range`, `--speed`, `--trace_len`, `--steps` |

## 9. Known constraints and pitfalls

- `reset()` is reset-free. It clears reward and episode counters, but it does not rebuild `self.state` or `self.Xfirst` from the original initialization. See [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):421-449.
- `calculate_v.py` has import-time side effects. Missing `.pt` files will break environment import before training starts.
- `visualize_self_propel.py` depends on the same environment-side `.pt` preprocessing files as training, because it imports `swimmer.py`, which imports `calculate_v.py`.
- `STOKES_NUM_THREADS` is set in `train.py` before importing `swimmer.py`, specifically so `calculate_v.py` picks it up during import.
- policy output path is timestamped `policy_<timestamp>`, not the older fixed `policy/` layout used elsewhere.
- `traj`, `traj2`, and `trajp` are flushed only every 4000 steps. If training stops early, in-memory data since the last flush will not be written automatically.
- the direction-stability penalty now starts only after 61 centroid samples are available, because it compares two equal 30-step windows.
- `_get_obs()` still references `self.reach_targets`, but this attribute is not initialized anywhere in this branch's `__init__`. This method looks stale and should not be trusted without inspection. See [`swimmer.py`](/F:/fyp/STOKES/RL_microrobots-master331/primitive_policies/flagella_self_propel/swimmer.py):451-452.
- there are many commented blocks from earlier experiments. Prefer tracing active logic from `train.py -> swimmer.py.step() -> calculate_v.py.RK()`.

## 10. Minimal reading order for future tasks

When starting a new task in this branch, read in this order unless the task is highly specific:

1. this file
2. `train.py`
3. `swimmer.py`
4. the relevant region of `calculate_v.py`
5. `discretization.py` only if the task touches geometry preprocessing or missing `.pt` assets

## 11. One-line mental model

PPO emits hinge-rate actions -> `swimmer_gym.step()` validates and scores them -> `RK()` integrates one macro-step using repeated Stokes-flow solves -> pressure-derived reward and direction-stability penalty shape the learned forward-propulsion policy.
