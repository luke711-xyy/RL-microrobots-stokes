import math
import os
from collections import deque
from pathlib import Path

import numpy as np
from gym import spaces
from gym.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import Policy

from calculate_v import NL as PRIMITIVE_LINK_NUM, RK_dual


directory_path = os.getcwd()

DT = 0.01
ENV_LINK_NUM = PRIMITIVE_LINK_NUM

ACTION_LOW = -1
ACTION_HIGH = 1

LOW_LEVEL_HOLD_STEPS = 25
MACRO_HORIZON = 50

GOAL_POINT = np.array([8.0, 0.0], dtype=np.float64)
GOAL_RADIUS = 0.3
NAV_PROGRESS_REWARD_COEF = 50.0
NAV_ANGLE_PENALTY_COEF = 10.0
NAV_REACH_BONUS = 20.0

ROBOT1_INIT = (4.0, -0.3)
ROBOT2_INIT = (4.0, 0.3)

ROBOT_IDS = ("robot_1", "robot_2")
PRIMITIVE_NAMES = ("forward", "cw", "ccw")
PRIMITIVE_TO_ID = {name: idx for idx, name in enumerate(PRIMITIVE_NAMES)}
PRIMITIVE_ID_TO_NAME = {idx: name for name, idx in PRIMITIVE_TO_ID.items()}
OBSERVATION_DIM = 8

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


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def is_checkpoint_path(path_obj):
    path_obj = Path(path_obj)
    if not path_obj.exists():
        return False
    if path_obj.is_file():
        return path_obj.name.startswith("checkpoint-")
    return (
        path_obj.name.startswith("checkpoint_")
        or (path_obj / "rllib_checkpoint.json").exists()
        or (path_obj / ".is_checkpoint").exists()
        or (path_obj / "policies" / "default_policy").exists()
    )


def checkpoint_sort_key(path_obj):
    path_obj = Path(path_obj)
    digits = "".join(ch for ch in path_obj.name if ch.isdigit())
    order = int(digits) if digits else -1
    return (order, str(path_obj))


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
        key=checkpoint_sort_key,
    )
    if direct_candidates:
        return direct_candidates[-1]

    raise FileNotFoundError(f"No RLlib policy directory found under: {path_obj}")


def restore_policy(path_str):
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


class swimmer_gym(MultiAgentEnv):
    metadata = {
        "render.modes": ["human"],
        "video.frames_per_second": 30,
    }

    def __init__(self, env_config):
        super().__init__()
        env_config = env_config or {}

        self.dt = DT
        self.low_level_hold_steps = int(env_config.get("low_level_hold_steps", LOW_LEVEL_HOLD_STEPS))
        self.macro_horizon = int(env_config.get("macro_horizon", MACRO_HORIZON))
        self.skip_policy_load = bool(env_config.get("skip_policy_load", False))
        self.forward_ckpt = env_config.get("forward_ckpt")
        self.cw_ckpt = env_config.get("cw_ckpt")
        self.ccw_ckpt = env_config.get("ccw_ckpt")

        self.betamax = (2 * math.pi) / ENV_LINK_NUM
        self.betamin = -self.betamax * 0.5

        self.action_space = spaces.Discrete(len(PRIMITIVE_NAMES))
        self.observation_space = spaces.Box(low=-10000, high=10000, shape=(OBSERVATION_DIM,), dtype=np.float64)

        self.low_level_policies = {}
        self.low_level_states = [{}, {}]
        if not self.skip_policy_load:
            self._load_low_level_policies()

        self.episode_count = 0
        self.it = 0
        self.low_level_step_count = 0
        self.ep_step = 0
        self.reward = 0.0
        self.done = False

        self.last_robot_rewards = [0.0, 0.0]
        self.last_robot_progress_rewards = [0.0, 0.0]
        self.last_robot_angle_penalties = [0.0, 0.0]
        self.last_robot_angle_errors = [0.0, 0.0]
        self.last_robot_goal_distances = [0.0, 0.0]
        self.last_robot_headings = [0.0, 0.0]
        self.last_robot_reached = [False, False]
        self.robot_goal_reached = [False, False]
        self.last_macro_action = (0, 0)
        self.last_macro_action_names = ("forward", "forward")
        self.last_centroid1 = np.zeros((2,), dtype=np.float64)
        self.last_centroid2 = np.zeros((2,), dtype=np.float64)
        self.last_substep_frames = []

        self.trace1 = deque(maxlen=1000)
        self.trace2 = deque(maxlen=1000)

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

        x_positions = np.zeros((ENV_LINK_NUM + 1,), dtype=np.float64)
        y_positions = np.zeros((ENV_LINK_NUM + 1,), dtype=np.float64)
        for i in range(ENV_LINK_NUM + 1):
            x_positions[i] = x_first[0] + i / ENV_LINK_NUM * math.cos(state[2])
            y_positions[i] = x_first[1] + i / ENV_LINK_NUM * math.sin(state[2])
        xy_positions = np.concatenate((x_positions.reshape(-1, 1), y_positions.reshape(-1, 1)), axis=1)

        true_centroid = compute_true_centroid(xy_positions)
        state[0] = true_centroid[0]
        state[1] = true_centroid[1]
        return state, x_first, xy_positions

    def _build_initial_geometry(self):
        self.state1, self.Xfirst1, self.XY_positions1 = self._build_initial_robot_state(ROBOT1_INIT)
        self.state2, self.Xfirst2, self.XY_positions2 = self._build_initial_robot_state(ROBOT2_INIT)

        self.current_primitives = ["forward", "forward"]
        self._reset_policy_states()

        centroid1 = compute_true_centroid(self.XY_positions1)
        centroid2 = compute_true_centroid(self.XY_positions2)
        self.last_centroid1 = np.array(centroid1, dtype=np.float64)
        self.last_centroid2 = np.array(centroid2, dtype=np.float64)
        self.trace1.clear()
        self.trace2.clear()
        self.trace1.append(np.array(centroid1))
        self.trace2.append(np.array(centroid2))

        self.robot_goal_reached = [False, False]
        self.last_robot_rewards = [0.0, 0.0]
        self.last_robot_progress_rewards = [0.0, 0.0]
        self.last_robot_angle_penalties = [0.0, 0.0]
        self.last_robot_headings = [
            compute_average_heading(self.state1),
            compute_average_heading(self.state2),
        ]
        self.last_robot_goal_distances = [
            float(np.linalg.norm(GOAL_POINT - centroid1)),
            float(np.linalg.norm(GOAL_POINT - centroid2)),
        ]
        self.last_robot_angle_errors = [0.0, 0.0]
        self.last_robot_reached = [False, False]
        self.last_macro_action = (0, 0)
        self.last_macro_action_names = ("forward", "forward")
        self.last_substep_frames = [self._capture_substep_frame(0)]

    def _reset_policy_states(self):
        if self.skip_policy_load:
            self.low_level_states = [{}, {}]
            return

        # 两个机器人共享底层权重，但各自维护自己的 recurrent state。
        self.low_level_states = []
        for _robot_idx in range(2):
            robot_states = {}
            for primitive_name, policy in self.low_level_policies.items():
                robot_states[primitive_name] = get_policy_initial_state(policy)
            self.low_level_states.append(robot_states)

    def _load_low_level_policies(self):
        required = {
            "forward": self.forward_ckpt,
            "cw": self.cw_ckpt,
            "ccw": self.ccw_ckpt,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"Missing low-level checkpoint paths for: {', '.join(missing)}")

        for primitive_name, ckpt_path in required.items():
            self.low_level_policies[primitive_name] = restore_policy(ckpt_path)

    def _get_single_obs(self, robot_idx):
        self_centroid = compute_true_centroid(self.XY_positions1 if robot_idx == 0 else self.XY_positions2)
        heading = self.last_robot_headings[robot_idx]
        prev_primitive = self.current_primitives[robot_idx]
        goal_relative = GOAL_POINT - self_centroid

        obs = np.concatenate(
            (
                np.array([self_centroid[0], self_centroid[1], heading], dtype=np.float64),
                primitive_to_one_hot(prev_primitive),
                np.array([goal_relative[0], goal_relative[1]], dtype=np.float64),
            ),
            axis=0,
        )
        return obs.astype(np.float64)

    def _get_obs(self):
        return {
            ROBOT_IDS[0]: self._get_single_obs(0),
            ROBOT_IDS[1]: self._get_single_obs(1),
        }

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
        recurrent_state = self.low_level_states[robot_idx][primitive_name]
        obs = self.state1[3:].copy() if robot_idx == 0 else self.state2[3:].copy()

        try:
            action_output = policy.compute_single_action(obs, state=recurrent_state, explore=False)
        except TypeError:
            action_output = policy.compute_single_action(obs, state=recurrent_state)

        action, next_state = unpack_action_output(action_output, recurrent_state)
        self.low_level_states[robot_idx][primitive_name] = next_state
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        target_state = self.state1 if robot_idx == 0 else self.state2
        return self._sanitize_low_level_action(target_state, action)

    def _apply_dual_solver(self, action1, action2):
        (
            state1_next,
            _xn1,
            _yn1,
            _r1,
            x_first_delta1,
            x_positions1,
            y_positions1,
            state2_next,
            _xn2,
            _yn2,
            _r2,
            x_first_delta2,
            x_positions2,
            y_positions2,
            _pressure_diff,
            _pressure_end,
            _pressure_all,
        ) = RK_dual(self.state1, action1, self.Xfirst1, self.state2, action2, self.Xfirst2)

        self.state1 = state1_next.copy()
        self.state2 = state2_next.copy()
        self.Xfirst1 = self.Xfirst1 + x_first_delta1
        self.Xfirst2 = self.Xfirst2 + x_first_delta2
        self.XY_positions1 = np.concatenate((np.array(x_positions1).reshape(-1, 1), np.array(y_positions1).reshape(-1, 1)), axis=1)
        self.XY_positions2 = np.concatenate((np.array(x_positions2).reshape(-1, 1), np.array(y_positions2).reshape(-1, 1)), axis=1)

        centroid1 = compute_true_centroid(self.XY_positions1)
        centroid2 = compute_true_centroid(self.XY_positions2)
        self.state1[0] = centroid1[0]
        self.state1[1] = centroid1[1]
        self.state2[0] = centroid2[0]
        self.state2[1] = centroid2[1]

    def _capture_substep_frame(self, substep_index):
        centroid1 = compute_true_centroid(self.XY_positions1)
        centroid2 = compute_true_centroid(self.XY_positions2)
        return {
            "substep_index": int(substep_index),
            "xy1": np.array(self.XY_positions1, copy=True),
            "xy2": np.array(self.XY_positions2, copy=True),
            "state1": np.array(self.state1, copy=True),
            "state2": np.array(self.state2, copy=True),
            "centroid1": np.array(centroid1, copy=True),
            "centroid2": np.array(centroid2, copy=True),
        }

    def _decode_action_dict(self, action_dict):
        decoded = []
        for robot_idx, robot_id in enumerate(ROBOT_IDS):
            action_id = int(action_dict.get(robot_id, PRIMITIVE_TO_ID[self.current_primitives[robot_idx]]))
            action_id = int(np.clip(action_id, 0, len(PRIMITIVE_NAMES) - 1))
            decoded.append((action_id, PRIMITIVE_ID_TO_NAME[action_id]))
        return decoded

    def _compute_navigation_reward(self, robot_idx, centroid_start, centroid_end, heading):
        goal_vec = GOAL_POINT - centroid_end
        goal_dist = float(np.linalg.norm(goal_vec))
        already_reached = self.robot_goal_reached[robot_idx]
        reached_now = goal_dist <= GOAL_RADIUS

        progress_reward = 0.0
        angle_penalty = 0.0
        angle_error = 0.0
        reach_bonus = 0.0

        if not reached_now:
            goal_unit = goal_vec / max(goal_dist, 1e-12)
            target_angle = math.atan2(goal_vec[1], goal_vec[0])
            angle_error = float(wrap_to_pi(heading - target_angle))
            progress_reward = NAV_PROGRESS_REWARD_COEF * float(np.dot(centroid_end - centroid_start, goal_unit))
            angle_penalty = -NAV_ANGLE_PENALTY_COEF * ((angle_error / math.pi) ** 2)
        elif not already_reached:
            reach_bonus = NAV_REACH_BONUS

        reward = progress_reward + angle_penalty + reach_bonus
        self.robot_goal_reached[robot_idx] = already_reached or reached_now
        return {
            "reward": float(reward),
            "progress_reward": float(progress_reward),
            "angle_penalty": float(angle_penalty),
            "angle_error": float(angle_error),
            "goal_distance": float(goal_dist),
            "heading": float(heading),
            "reached": bool(self.robot_goal_reached[robot_idx]),
        }

    def _record_macro_step(self, robot_rewards):
        global traj
        global traj2
        global trajp

        combined_state = np.concatenate((self.state1.copy(), self.state2.copy()), axis=0)
        summary_row = np.array(
            [
                self.state1[0],
                self.state1[1],
                self.state2[0],
                self.state2[1],
                self.last_robot_rewards[0],
                self.last_robot_rewards[1],
                float(self.last_macro_action[0]),
                float(self.last_macro_action[1]),
                self.last_robot_goal_distances[0],
                self.last_robot_goal_distances[1],
            ],
            dtype=np.float64,
        )
        reward_row = np.array(
            [
                self.last_robot_progress_rewards[0],
                self.last_robot_progress_rewards[1],
                self.last_robot_angle_penalties[0],
                self.last_robot_angle_penalties[1],
                self.last_robot_angle_errors[0],
                self.last_robot_angle_errors[1],
                float(self.last_robot_reached[0]),
                float(self.last_robot_reached[1]),
            ],
            dtype=np.float64,
        )

        traj = _stack_trace(traj, combined_state)
        traj2 = _stack_trace(traj2, summary_row)
        trajp = _stack_trace(trajp, reward_row)

        if self.ep_step > 0 and self.ep_step % 100 == 0:
            path1 = os.path.join(directory_path, "traj")
            path2 = os.path.join(directory_path, "traj2")
            pathp = os.path.join(directory_path, "trajp")
            os.makedirs(path1, exist_ok=True)
            os.makedirs(path2, exist_ok=True)
            os.makedirs(pathp, exist_ok=True)

            np.savetxt(os.path.join(path1, f"traj_{len(os.listdir(path1))}.pt"), traj, delimiter=",")
            np.savetxt(os.path.join(path2, f"traj2_{len(os.listdir(path2))}.pt"), traj2, delimiter=",")
            np.savetxt(os.path.join(pathp, f"trajp_{len(os.listdir(pathp))}.pt"), trajp, delimiter=",")

            traj = []
            traj2 = []
            trajp = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action_dict):
        self.it += 1
        self.ep_step += 1
        self.reward = 0.0
        self.done = False

        decoded_actions = self._decode_action_dict(action_dict or {})
        self.last_macro_action = (decoded_actions[0][0], decoded_actions[1][0])
        self.last_macro_action_names = (decoded_actions[0][1], decoded_actions[1][1])
        self.current_primitives = [decoded_actions[0][1], decoded_actions[1][1]]

        centroid1_start = compute_true_centroid(self.XY_positions1)
        centroid2_start = compute_true_centroid(self.XY_positions2)
        self.last_substep_frames = []

        for substep_index in range(self.low_level_hold_steps):
            action1 = self._compute_low_level_action(0, self.current_primitives[0])
            action2 = self._compute_low_level_action(1, self.current_primitives[1])
            self._apply_dual_solver(action1, action2)
            self.low_level_step_count += 1
            self.last_substep_frames.append(self._capture_substep_frame(substep_index + 1))

        centroid1_end = compute_true_centroid(self.XY_positions1)
        centroid2_end = compute_true_centroid(self.XY_positions2)
        heading1 = compute_average_heading(self.state1)
        heading2 = compute_average_heading(self.state2)

        self.last_centroid1 = np.array(centroid1_end, dtype=np.float64)
        self.last_centroid2 = np.array(centroid2_end, dtype=np.float64)
        self.trace1.append(np.array(centroid1_end))
        self.trace2.append(np.array(centroid2_end))

        nav1 = self._compute_navigation_reward(0, centroid1_start, centroid1_end, heading1)
        nav2 = self._compute_navigation_reward(1, centroid2_start, centroid2_end, heading2)

        self.last_robot_rewards = [nav1["reward"], nav2["reward"]]
        self.last_robot_progress_rewards = [nav1["progress_reward"], nav2["progress_reward"]]
        self.last_robot_angle_penalties = [nav1["angle_penalty"], nav2["angle_penalty"]]
        self.last_robot_angle_errors = [nav1["angle_error"], nav2["angle_error"]]
        self.last_robot_goal_distances = [nav1["goal_distance"], nav2["goal_distance"]]
        self.last_robot_headings = [nav1["heading"], nav2["heading"]]
        self.last_robot_reached = [nav1["reached"], nav2["reached"]]

        print(
            f"[Macro {self.ep_step:>3d}] pair={self.current_primitives[0]}-{self.current_primitives[1]} | "
            f"R1: reward={nav1['reward']:>8.4f}, prog={nav1['progress_reward']:>8.4f}, "
            f"ang_pen={nav1['angle_penalty']:>8.4f}, ang_err={nav1['angle_error']:>7.4f}, "
            f"dist={nav1['goal_distance']:>7.4f}, reached={int(nav1['reached'])} | "
            f"R2: reward={nav2['reward']:>8.4f}, prog={nav2['progress_reward']:>8.4f}, "
            f"ang_pen={nav2['angle_penalty']:>8.4f}, ang_err={nav2['angle_error']:>7.4f}, "
            f"dist={nav2['goal_distance']:>7.4f}, reached={int(nav2['reached'])}"
        )

        self._record_macro_step(self.last_robot_rewards)

        if all(self.robot_goal_reached):
            self.done = True
        elif self.ep_step >= self.macro_horizon:
            self.done = True

        obs = self._get_obs()
        rewards = {
            ROBOT_IDS[0]: float(nav1["reward"]),
            ROBOT_IDS[1]: float(nav2["reward"]),
        }
        dones = {robot_id: self.done for robot_id in ROBOT_IDS}
        dones["__all__"] = self.done
        infos = {
            ROBOT_IDS[0]: {
                "reward": float(nav1["reward"]),
                "progress_reward": float(nav1["progress_reward"]),
                "angle_penalty": float(nav1["angle_penalty"]),
                "angle_error": float(nav1["angle_error"]),
                "goal_distance": float(nav1["goal_distance"]),
                "heading": float(nav1["heading"]),
                "reached": bool(nav1["reached"]),
                "primitive": self.current_primitives[0],
            },
            ROBOT_IDS[1]: {
                "reward": float(nav2["reward"]),
                "progress_reward": float(nav2["progress_reward"]),
                "angle_penalty": float(nav2["angle_penalty"]),
                "angle_error": float(nav2["angle_error"]),
                "goal_distance": float(nav2["goal_distance"]),
                "heading": float(nav2["heading"]),
                "reached": bool(nav2["reached"]),
                "primitive": self.current_primitives[1],
            },
        }
        return obs, rewards, dones, infos

    def reset(self):
        self._build_initial_geometry()
        self.reward = 0.0
        self.done = False
        self.ep_step = 0
        self.episode_count += 1
        return self._get_obs()

    def render(self):
        return None
