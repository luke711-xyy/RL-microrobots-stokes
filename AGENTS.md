# AGENTS.md

This file records the working conventions for this repository so future edits stay consistent with the current project setup.

## 1. Repository purpose

This repository implements hierarchical reinforcement learning for microrobot chemotaxis and primitive swimming policy acquisition.

The currently active low-level work is centered on:

- `primitive_policies/flagella_self_propel`
- `primitive_policies/flagella_turn`

Unless a task explicitly says otherwise, treat these two branches as the maintained low-level flagellar policy code paths.

## 2. Directory conventions

### Primitive-policy branches

Each primitive-policy branch should keep the same basic runtime skeleton:

- `discretization.py`
- `calculate_v.py`
- `swimmer.py`
- `train.py`
- optional visualizer such as `visualize_self_propel.py` or `visualize_turn.py`

When creating a new primitive-policy branch, prefer copying the closest maintained branch and making the minimum task-specific edits instead of redesigning the structure.

### Current branch roles

- `flagella_self_propel`: forward propulsion primitive
- `flagella_turn`: directional turning primitive with `cw` / `ccw` target switching

## 3. Runtime pipeline

For low-level flagellar branches, the expected pipeline is:

1. Run `discretization.py`
2. This generates the `.pt` preprocessing artifacts required by `calculate_v.py`
3. Run `train.py`
4. `train.py` imports `swimmer.py`
5. `swimmer.py` imports `calculate_v.py`
6. `calculate_v.py` loads the `.pt` files at import time

Important:

- The `.pt` preprocessing files are hard prerequisites, not optional cache files.
- If they are missing in the branch working directory, training or visualization will fail during import.

## 4. Training conventions

### Parameter recording

Every training run must create a Markdown record file at the root of the generated policy directory:

- `TRAINING_PARAMS.md`

This file should include:

- run timestamp
- output policy directory
- CLI arguments
- PPO / RLlib training parameters
- key environment parameters
- reward coefficients and window definitions

Do not remove this behavior from training entrypoints.

### Output directories

Self-propel branch:

- output root: `policy_<timestamp>`

Turn branch:

- output root: `policy_cw_<timestamp>`
- output root: `policy_ccw_<timestamp>`

Checkpoint directories should remain RLlib-compatible and continue to be saved underneath numeric iteration folders such as:

- `policy_.../0/`
- `policy_.../10/`
- `policy_.../20/`

### Auxiliary outputs

Training branches may also create:

- `traj/`
- `traj2/`
- `trajp/`

Preserve this layout unless a task explicitly changes the trajectory storage format.

## 5. Environment conventions

### True centroid

In maintained low-level flagellar branches, “centroid” means the true geometric centroid computed from `XY_positions`.

Do not mix:

- translational state variables
- geometric centroid derived from body points

If reward, trajectory logging, and visualization all refer to centroid, they should all use the same geometric definition.

### Reset behavior

Current maintained low-level flagellar branches use reset-free behavior across episode boundaries unless a task explicitly changes that design.

If reset behavior changes, document it clearly in both code comments and the training parameter record.

### Reward shaping

Keep reward logic localized in `swimmer.py`.

When changing reward design:

- preserve pressure-reward and directional / turning reward terms as separately inspectable components
- keep debug fields such as `last_pressure_reward` and branch-specific diagnostics available for training logs and visualizers
- keep reward coefficients explicit constants or explicit environment attributes, not hidden magic numbers deep in expressions

## 6. Visualizer conventions

Visualizers should stay structurally close to the maintained reference scripts already in the repository.

Expected behavior:

- support `--num_cpus`
- support `--num_threads`
- support explicit `--checkpoint`
- support latest-checkpoint auto-discovery when appropriate
- render swimmer body, centroid, centroid trace, and average-heading cue
- show branch-specific reward diagnostics in the info panel and terminal output

For macOS compatibility:

- parse CLI args before importing the environment if thread env vars are needed
- avoid introducing extra rendering layers unless the task explicitly calls for them

## 7. Branch-specific expectations

### `primitive_policies/flagella_self_propel`

Use `primitive_policies/flagella_self_propel/CODE_INDEX.md` as the first reference document before making substantial edits in this branch.

Append meaningful progress to:

- `primitive_policies/flagella_self_propel/WORKLOG.md`

when tasks materially change behavior, reward logic, training configuration, visualization, or runtime assumptions.

### `primitive_policies/flagella_turn`

This branch is intentionally parallel to `flagella_self_propel` and should stay structurally aligned with it unless there is a strong task-specific reason to diverge.

Current task identity:

- turning target supplied via `--turn_direction {cw,ccw}`
- turning reward based on the signed angle between two 100-step true-centroid displacement windows
- pressure reward and turning reward both contribute to total reward

## 8. Git and artifact hygiene

Do not commit:

- `__pycache__/`
- generated `.pyc` files
- transient local training artifacts unless explicitly requested

Commit only source changes and intentional documentation changes.

When a task introduces a new maintained behavior or workflow, update the closest relevant documentation instead of relying only on chat history.

## 9. Practical defaults for future work

Unless a task says otherwise:

- prefer minimal structural changes
- preserve existing file layout and naming
- preserve RLlib checkpoint compatibility
- preserve command-line configurability for CPU/thread counts
- preserve train/visualize config parity within the same branch

When in doubt, copy the nearest maintained branch and change only the task-defining parts:

- reward
- branch-specific CLI arguments
- output naming
- visualizer labels
