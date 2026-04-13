import math
from collections import deque
from itertools import product
from pathlib import Path

import numpy as np

from calculate_v import NL as PRIMITIVE_LINK_NUM, RK_triple

try:
    import gym
    from gym import spaces
    from gym.utils import seeding
except ModuleNotFoundError:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.utils import seeding

try:
    from ray.rllib.policy.policy import Policy
except ModuleNotFoundError:
    Policy = None


BASE_DIR = Path(__file__).resolve().parent

MAX_STEP = 10000
DT = 0.01
ENV_LINK_NUM = PRIMITIVE_LINK_NUM

ACTION_LOW = -1
ACTION_HIGH = 1

LOW_LEVEL_HOLD_STEPS = 25
MACRO_HORIZON = 200

FORWARD_REWARD_COEF = 1.0
DELTA_X12_PENALTY_COEF = 1.0
DELTA_X23_PENALTY_COEF = 1.0
DELTA_Y12_PENALTY_COEF = 2.0
DELTA_Y23_PENALTY_COEF = 2.0

FORMATION_TARGET_DX12 = 0.0
FORMATION_TARGET_DX23 = 0.0
FORMATION_TARGET_DY12 = 1.0
FORMATION_TARGET_DY23 = 1.0

ROBOT1_INIT = (-4.0, 1.0)
ROBOT2_INIT = (-4.0, 0.0)
ROBOT3_INIT = (-4.0, -1.0)
ROBOT_INITS = (ROBOT1_INIT, ROBOT2_INIT, ROBOT3_INIT)

PRIMITIVE_NAMES = ("forward", "cw", "ccw")
PRIMITIVE_TO_ID = {name: idx for idx, name in enumerate(PRIMITIVE_NAMES)}
MACRO_ACTION_TABLE = list(product(PRIMITIVE_NAMES, repeat=3))


traj = []
traj2 = []
trajp = []


def _stack_trace(existing, row):
    row = np.asarray(row, dtype=np.float64).reshape(1, -1)
    if isinstance(existing, list) and len(existing) == 0:
        return row
    return np.concatenate((existing.reshape(-1, row.shape[1]), row), axis=0)


def compute_true_centroid(xy_positions):
    xy_positions = np.asarray(xy_positions, dtype=np.float64)
    return np.mean(xy_positions, axis=0)


def compute_average_heading(state_array):
    head_omega = state_array[2]
    running_angle = head_omega
    angle_sum = head_omega
    for beta in state_array[3:]:
        running_angle += beta
        angle_sum += running_angle
    return angle_sum / (len(state_array) - 2)


def primitive_to_one_hot(primitive_name):
    one_hot = np.zeros((len(PRIMITIVE_NAMES),), dtype=np.float64)
    one_hot[PRIMITIVE_TO_ID[primitive_name]] = 1.0
    return one_hot


def resolve_policy_checkpoint_dir(path_str):
    path_obj = Path(path_str).expanduser().resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path_obj}")

    if path_obj.is_file():
        candidate = path_obj.parent / "policies" / "default_policy"
        if candidate.exists():
            return candidate

    if (path_obj / "policies" / "default_policy").exists():
        return path_obj / "policies" / "default_policy"

    direct_candidates = sorted(
        [candidate for candidate in path_obj.rglob("default_policy") if candidate.is_dir() and candidate.name == "default_policy"],
        key=lambda item: (int("".join(ch for ch in item.name if ch.isdigit()) or "-1"), str(item)),
    )
    if direct_candidates:
        return direct_candidates[-1]

    raise FileNotFoundError(f"No RLlib policy directory found under: {path_obj}")


def restore_policy(path_str):
    if Policy is None:
        raise ModuleNotFoundError("ray[rllib] is required to restore primitive checkpoints")
    policy_dir = resolve_policy_checkpoint_dir(path_str)
    restored = Policy.from_checkpoint(str(policy_dir))
    if isinstance(restored, dict):
        if "default_policy" in restored:
            return restored["default_policy"]
        if len(restored) == 1:
            return next(iter(restored.values()))
        raise ValueError(f"Unexpected policy dictionary keys: {list(restored.keys())}")
    return restored


def get_policy_initial_state(policy):
    try:
        state = policy.get_initial_state()
    except Exception:
        return []
    return [np.array(item, copy=True) for item in state]


def unpack_action_output(action_output, prev_state):
    if not isinstance(action_output, tuple):
        return action_output, prev_state
    if len(action_output) == 0:
        raise ValueError("compute_single_action returned an empty tuple")
    action = action_output[0]
    next_state = prev_state
    if len(action_output) >= 2 and isinstance(action_output[1], (list, tuple)):
        next_state = action_output[1]
    return action, next_state


class swimmer_gym(gym.Env):
    metadata = {"render.modes": ["human"], "video.frames_per_second": 30}

    def __init__(self, env_config):
        env_config = env_config or {}

        self.dt = DT
        self.low_level_hold_steps = int(env_config.get("low_level_hold_steps", LOW_LEVEL_HOLD_STEPS))
        self.macro_horizon = int(env_config.get("macro_horizon", MACRO_HORIZON))
        self.skip_policy_load = bool(env_config.get("skip_policy_load", False))
        self.forward_ckpt = env_config.get("forward_ckpt")
        self.cw_ckpt = env_config.get("cw_ckpt")
        self.ccw_ckpt = env_config.get("ccw_ckpt")

        self.betamax = (2 * math.pi) / ENV_LINK_NUM

        self.action_space = spaces.Discrete(len(MACRO_ACTION_TABLE))
        self.observation_space = spaces.Box(low=-10000, high=10000, shape=(22,), dtype=np.float64)
        self.viewer = None

        self.low_level_policies = {}
        self.low_level_states = [{} for _ in ROBOT_INITS]
        if not self.skip_policy_load:
            self._load_low_level_policies()

        self.episode_count = 0
        self.it = 0
        self.low_level_step_count = 0
        self.ep_step = 0
        self.reward = 0.0
        self.done = False

        self.last_forward_reward = 0.0
        self.last_dx12_penalty = 0.0
        self.last_dx23_penalty = 0.0
        self.last_dy12_penalty = 0.0
        self.last_dy23_penalty = 0.0
        self.last_dx12 = 0.0
        self.last_dx23 = 0.0
        self.last_dy12 = 0.0
        self.last_dy23 = 0.0
        self.last_macro_action = 0
        self.last_macro_action_names = MACRO_ACTION_TABLE[0]
        self.last_centroid1 = np.zeros((2,), dtype=np.float64)
        self.last_centroid2 = np.zeros((2,), dtype=np.float64)
        self.last_centroid3 = np.zeros((2,), dtype=np.float64)
        self.last_substep_frames = []

        self.trace1 = deque(maxlen=1000)
        self.trace2 = deque(maxlen=1000)
        self.trace3 = deque(maxlen=1000)

        self._build_initial_geometry()

    def _build_initial_robot_state(self, init_xy):
        centroid_x, centroid_y = init_xy
        state = np.zeros((ENV_LINK_NUM + 2,), dtype=np.float64)
        state[0] = centroid_x
        state[1] = centroid_y
        state[2] = 0.0

        x_first = np.zeros((2,), dtype=np.float64)
        x_first[0] = centroid_x - 0.5 * math.cos(state[2])
        x_first[1] = centroid_y - 0.5 * math.sin(state[2])

        Xp = np.zeros((ENV_LINK_NUM + 1,), dtype=np.float64)
        Yp = np.zeros((ENV_LINK_NUM + 1,), dtype=np.float64)
        for i in range(ENV_LINK_NUM + 1):
            Xp[i] = x_first[0] + i / ENV_LINK_NUM * math.cos(state[2])
            Yp[i] = x_first[1] + i / ENV_LINK_NUM * math.sin(state[2])
        XY_positions = np.concatenate((Xp.reshape(-1, 1), Yp.reshape(-1, 1)), axis=1)

        true_centroid = compute_true_centroid(XY_positions)
        state[0] = true_centroid[0]
        state[1] = true_centroid[1]
        return state, x_first, XY_positions

    def _build_initial_geometry(self):
        self.states = []
        self.Xfirsts = []
        self.XY_positions = []
        for init_xy in ROBOT_INITS:
            state, x_first, xy_positions = self._build_initial_robot_state(init_xy)
            self.states.append(state)
            self.Xfirsts.append(x_first)
            self.XY_positions.append(xy_positions)

        self.current_primitives = ["forward"] * len(ROBOT_INITS)
        self._reset_policy_states()

        centroids = [compute_true_centroid(xy_positions) for xy_positions in self.XY_positions]
        self.last_centroid1 = np.array(centroids[0], dtype=np.float64)
        self.last_centroid2 = np.array(centroids[1], dtype=np.float64)
        self.last_centroid3 = np.array(centroids[2], dtype=np.float64)
        self.trace1.clear()
        self.trace2.clear()
        self.trace3.clear()
        self.trace1.append(np.array(centroids[0]))
        self.trace2.append(np.array(centroids[1]))
        self.trace3.append(np.array(centroids[2]))

    def _reset_policy_states(self):
        if self.skip_policy_load:
            self.low_level_states = [{} for _ in ROBOT_INITS]
            return

        self.low_level_states = []
        for _ in ROBOT_INITS:
            robot_states = {}
            for primitive_name, policy in self.low_level_policies.items():
                robot_states[primitive_name] = get_policy_initial_state(policy)
            self.low_level_states.append(robot_states)

    def _load_low_level_policies(self):
        required = {"forward": self.forward_ckpt, "cw": self.cw_ckpt, "ccw": self.ccw_ckpt}
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"Missing low-level checkpoint paths for: {', '.join(missing)}")

        for primitive_name, ckpt_path in required.items():
            self.low_level_policies[primitive_name] = restore_policy(ckpt_path)

    def _get_obs(self):
        obs_parts = []
        centroids = [compute_true_centroid(xy_positions) for xy_positions in self.XY_positions]
        for robot_idx, centroid in enumerate(centroids):
            avg_heading = compute_average_heading(self.states[robot_idx])
            obs_parts.append(np.array([centroid[0], centroid[1], avg_heading], dtype=np.float64))
            obs_parts.append(primitive_to_one_hot(self.current_primitives[robot_idx]))

        dx12 = centroids[0][0] - centroids[1][0]
        dy12 = centroids[0][1] - centroids[1][1]
        dx23 = centroids[1][0] - centroids[2][0]
        dy23 = centroids[1][1] - centroids[2][1]
        obs_parts.append(np.array([dx12, dy12, dx23, dy23], dtype=np.float64))
        return np.concatenate(obs_parts, axis=0).astype(np.float64)

    def _sanitize_low_level_action(self, state, action):
        action = np.asarray(action, dtype=np.float64)
        clipped = np.clip(action, ACTION_LOW, ACTION_HIGH)
        state_predict = state.copy()
        state_predict[3:] += clipped * 0.2
        if np.any(np.abs(state_predict[3:]) > self.betamax):
            return np.zeros_like(clipped)
        return clipped

    def _compute_low_level_action(self, robot_idx, primitive_name):
        if self.skip_policy_load:
            return np.zeros((ENV_LINK_NUM - 1,), dtype=np.float64)

        policy = self.low_level_policies[primitive_name]
        state = self.low_level_states[robot_idx][primitive_name]
        obs = self.states[robot_idx][3:].copy()

        try:
            action_output = policy.compute_single_action(obs, state=state, explore=False)
        except TypeError:
            action_output = policy.compute_single_action(obs, state=state)

        action, next_state = unpack_action_output(action_output, state)
        self.low_level_states[robot_idx][primitive_name] = next_state
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        return self._sanitize_low_level_action(self.states[robot_idx], action)

    def _apply_triple_solver(self, actions):
        (
            state1_next,
            _Xn1,
            _Yn1,
            _r1,
            x_first_delta1,
            Xpositions1,
            Ypositions1,
            state2_next,
            _Xn2,
            _Yn2,
            _r2,
            x_first_delta2,
            Xpositions2,
            Ypositions2,
            state3_next,
            _Xn3,
            _Yn3,
            _r3,
            x_first_delta3,
            Xpositions3,
            Ypositions3,
            _pressure_diff,
            _pressure_end,
            _pressure_all,
        ) = RK_triple(
            self.states[0],
            actions[0],
            self.Xfirsts[0],
            self.states[1],
            actions[1],
            self.Xfirsts[1],
            self.states[2],
            actions[2],
            self.Xfirsts[2],
        )

        next_states = [state1_next.copy(), state2_next.copy(), state3_next.copy()]
        x_first_deltas = [x_first_delta1, x_first_delta2, x_first_delta3]
        positions_x = [Xpositions1, Xpositions2, Xpositions3]
        positions_y = [Ypositions1, Ypositions2, Ypositions3]

        for robot_idx in range(3):
            self.states[robot_idx] = next_states[robot_idx]
            self.Xfirsts[robot_idx] = self.Xfirsts[robot_idx] + x_first_deltas[robot_idx]
            self.XY_positions[robot_idx] = np.concatenate(
                (
                    np.array(positions_x[robot_idx]).reshape(-1, 1),
                    np.array(positions_y[robot_idx]).reshape(-1, 1),
                ),
                axis=1,
            )
            centroid = compute_true_centroid(self.XY_positions[robot_idx])
            self.states[robot_idx][0] = centroid[0]
            self.states[robot_idx][1] = centroid[1]

    def _capture_substep_frame(self, substep_index):
        centroids = [compute_true_centroid(xy_positions) for xy_positions in self.XY_positions]
        return {
            "substep_index": int(substep_index),
            "xy1": np.array(self.XY_positions[0], copy=True),
            "xy2": np.array(self.XY_positions[1], copy=True),
            "xy3": np.array(self.XY_positions[2], copy=True),
            "state1": np.array(self.states[0], copy=True),
            "state2": np.array(self.states[1], copy=True),
            "state3": np.array(self.states[2], copy=True),
            "centroid1": np.array(centroids[0], copy=True),
            "centroid2": np.array(centroids[1], copy=True),
            "centroid3": np.array(centroids[2], copy=True),
        }

    def _decode_macro_action(self, action):
        return MACRO_ACTION_TABLE[int(action)]

    def _record_macro_step(self, reward):
        global traj
        global traj2
        global trajp

        combined_state = np.concatenate([state.copy() for state in self.states], axis=0)
        summary_row = np.array(
            [
                self.states[0][0],
                self.states[0][1],
                self.states[1][0],
                self.states[1][1],
                self.states[2][0],
                self.states[2][1],
                self.last_dx12,
                self.last_dx23,
                self.last_dy12,
                self.last_dy23,
                self.last_forward_reward,
                reward,
                float(self.last_macro_action),
            ],
            dtype=np.float64,
        )
        reward_row = np.array(
            [
                self.last_forward_reward,
                self.last_dx12_penalty,
                self.last_dx23_penalty,
                self.last_dy12_penalty,
                self.last_dy23_penalty,
            ],
            dtype=np.float64,
        )

        traj = _stack_trace(traj, combined_state)
        traj2 = _stack_trace(traj2, summary_row)
        trajp = _stack_trace(trajp, reward_row)

        if self.ep_step > 0 and self.ep_step % 100 == 0:
            path1 = BASE_DIR / "traj"
            path2 = BASE_DIR / "traj2"
            pathp = BASE_DIR / "trajp"
            path1.mkdir(exist_ok=True)
            path2.mkdir(exist_ok=True)
            pathp.mkdir(exist_ok=True)

            np.savetxt(path1 / f"traj_{len(list(path1.iterdir()))}.pt", traj, delimiter=",")
            np.savetxt(path2 / f"traj2_{len(list(path2.iterdir()))}.pt", traj2, delimiter=",")
            np.savetxt(pathp / f"trajp_{len(list(pathp.iterdir()))}.pt", trajp, delimiter=",")

            traj = []
            traj2 = []
            trajp = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.it += 1
        self.ep_step += 1
        self.reward = 0.0
        self.done = False

        primitives = self._decode_macro_action(action)
        self.last_macro_action = int(action)
        self.last_macro_action_names = primitives
        self.current_primitives = list(primitives)

        centroids_start = [compute_true_centroid(xy_positions) for xy_positions in self.XY_positions]
        self.last_substep_frames = []

        for substep_index in range(self.low_level_hold_steps):
            actions = [self._compute_low_level_action(robot_idx, primitives[robot_idx]) for robot_idx in range(3)]
            self._apply_triple_solver(actions)
            self.low_level_step_count += 1
            self.last_substep_frames.append(self._capture_substep_frame(substep_index + 1))

        centroids_end = [compute_true_centroid(xy_positions) for xy_positions in self.XY_positions]
        self.last_centroid1 = np.array(centroids_end[0], dtype=np.float64)
        self.last_centroid2 = np.array(centroids_end[1], dtype=np.float64)
        self.last_centroid3 = np.array(centroids_end[2], dtype=np.float64)
        self.trace1.append(np.array(centroids_end[0]))
        self.trace2.append(np.array(centroids_end[1]))
        self.trace3.append(np.array(centroids_end[2]))

        dx1 = centroids_end[0][0] - centroids_start[0][0]
        dx2 = centroids_end[1][0] - centroids_start[1][0]
        dx3 = centroids_end[2][0] - centroids_start[2][0]
        self.last_forward_reward = FORWARD_REWARD_COEF * ((dx1 + dx2 + dx3) / 3.0)

        self.last_dx12 = centroids_end[0][0] - centroids_end[1][0]
        self.last_dx23 = centroids_end[1][0] - centroids_end[2][0]
        self.last_dy12 = centroids_end[0][1] - centroids_end[1][1]
        self.last_dy23 = centroids_end[1][1] - centroids_end[2][1]

        self.last_dx12_penalty = -DELTA_X12_PENALTY_COEF * abs(self.last_dx12 - FORMATION_TARGET_DX12)
        self.last_dx23_penalty = -DELTA_X23_PENALTY_COEF * abs(self.last_dx23 - FORMATION_TARGET_DX23)
        self.last_dy12_penalty = -DELTA_Y12_PENALTY_COEF * abs(self.last_dy12 - FORMATION_TARGET_DY12)
        self.last_dy23_penalty = -DELTA_Y23_PENALTY_COEF * abs(self.last_dy23 - FORMATION_TARGET_DY23)

        macro_reward = (
            self.last_forward_reward
            + self.last_dx12_penalty
            + self.last_dx23_penalty
            + self.last_dy12_penalty
            + self.last_dy23_penalty
        )
        self.reward += macro_reward

        print(
            f"[Macro {self.ep_step:>3d}] trio={primitives[0]}-{primitives[1]}-{primitives[2]} | "
            f"Reward: {macro_reward:>9.4f}, Forward: {self.last_forward_reward:>9.4f}, "
            f"Dx12Pen: {self.last_dx12_penalty:>9.4f}, Dx23Pen: {self.last_dx23_penalty:>9.4f}, "
            f"Dy12Pen: {self.last_dy12_penalty:>9.4f}, Dy23Pen: {self.last_dy23_penalty:>9.4f} | "
            f"R1: ({centroids_end[0][0]:>9.4f}, {centroids_end[0][1]:>9.4f}), "
            f"R2: ({centroids_end[1][0]:>9.4f}, {centroids_end[1][1]:>9.4f}), "
            f"R3: ({centroids_end[2][0]:>9.4f}, {centroids_end[2][1]:>9.4f}) | "
            f"dx12={self.last_dx12:>8.4f}, dx23={self.last_dx23:>8.4f}, "
            f"dy12={self.last_dy12:>8.4f}, dy23={self.last_dy23:>8.4f}"
        )

        self._record_macro_step(macro_reward)

        if self.ep_step >= self.macro_horizon:
            self.done = True

        return self._get_obs(), float(self.reward), self.done, {}

    def reset(self):
        self.reward = 0.0
        self.done = False
        self.ep_step = 0
        self.episode_count += 1
        self.last_substep_frames = [self._capture_substep_frame(0)]
        return self._get_obs()

    def render(self):
        return None
