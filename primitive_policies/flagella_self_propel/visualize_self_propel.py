import argparse
import math
import os
import sys
from collections import deque
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a trained flagella self-propel policy")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file or containing directory. Defaults to the latest local policy.")
    parser.add_argument("--steps", type=int, default=2000, help="Visualization steps to simulate.")
    parser.add_argument("--speed", type=float, default=0.03, help="Pause duration between frames in seconds.")
    parser.add_argument("--num_cpus", type=int, default=1, help="Number of CPUs for Ray during visualization.")
    parser.add_argument("--num_threads", type=int, default=5, help="Number of PyTorch threads for the Stokes solver.")
    parser.add_argument("--view_range", type=float, default=4.5, help="Half-width of the plotting window around the swimmer centroid.")
    parser.add_argument("--trace_len", type=int, default=300, help="Maximum stored centroid points in the path trace.")
    return parser.parse_args()


ARGS = parse_args()
os.environ["STOKES_NUM_THREADS"] = str(ARGS.num_threads)
os.chdir(BASE_DIR)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import ray
import ray.rllib.algorithms.ppo as ppo

from swimmer import swimmer_gym


def resolve_checkpoint(path_str):
    path = Path(path_str).expanduser().resolve()
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    if is_checkpoint_path(path):
        return path

    direct_candidates = sorted(
        [candidate for candidate in path.iterdir() if is_checkpoint_path(candidate)],
        key=checkpoint_sort_key,
    )
    if direct_candidates:
        return direct_candidates[-1]

    nested_candidates = sorted(
        [candidate for candidate in path.rglob("*") if is_checkpoint_path(candidate)],
        key=checkpoint_sort_key,
    )
    if nested_candidates:
        return nested_candidates[-1]

    raise FileNotFoundError(f"No checkpoint path found under: {path}")


def is_checkpoint_path(path):
    path = Path(path)
    if not path.exists():
        return False
    if path.is_file():
        return path.name.startswith("checkpoint-")
    return (
        path.name.startswith("checkpoint_")
        or (path / "rllib_checkpoint.json").exists()
        or (path / ".is_checkpoint").exists()
    )


def checkpoint_sort_key(path):
    path = Path(path)
    digits = "".join(ch for ch in path.name if ch.isdigit())
    order = int(digits) if digits else -1
    return (order, str(path))


def find_latest_checkpoint(base_dir=None):
    base_dir = Path(base_dir or BASE_DIR)
    policy_roots = [path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith("policy_")]
    if not policy_roots:
        return None

    latest_root = max(policy_roots, key=lambda p: p.stat().st_mtime)
    iter_dirs = [path for path in latest_root.iterdir() if path.is_dir() and path.name.isdigit()]
    if not iter_dirs:
        return None

    latest_iter_dir = max(iter_dirs, key=lambda p: int(p.name))
    checkpoint_paths = sorted(
        [candidate for candidate in latest_iter_dir.rglob("*") if is_checkpoint_path(candidate)],
        key=checkpoint_sort_key,
    )
    if not checkpoint_paths:
        return None

    return checkpoint_paths[-1]


def build_config():
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 0
    config["num_rollout_workers"] = 0
    config["framework"] = "torch"
    config["gamma"] = 0.9999
    config["lr"] = 0.0003
    config["horizon"] = 1000
    config["evaluation_duration"] = 10000000
    config["lr_schedule"] = None
    config["use_critic"] = True
    config["use_gae"] = True
    config["lambda_"] = 0.95
    config["kl_coeff"] = 0.2
    config["sgd_minibatch_size"] = 64
    config["train_batch_size"] = 1000
    config["num_sgd_iter"] = 30
    config["shuffle_sequences"] = True
    config["vf_loss_coeff"] = 1.0
    config["entropy_coeff"] = 0.0
    config["entropy_coeff_schedule"] = None
    config["clip_param"] = 0.1
    config["vf_clip_param"] = 100000
    config["grad_clip"] = None
    config["kl_target"] = 0.01
    config["evaluation_interval"] = 1000000
    config["evaluation_duration"] = 1
    config["use_lstm"] = True
    config["max_seq_len"] = 20
    config["min_sample_timesteps_per_iteration"] = 1000
    return config


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


def compute_average_heading(state):
    if len(state) <= 3:
        return float(state[2])

    heading = float(state[2])
    running_angle = heading
    angle_sum = heading
    for beta in state[3:]:
        running_angle += float(beta)
        angle_sum += running_angle
    return angle_sum / (len(state) - 2)


def main():
    checkpoint = resolve_checkpoint(ARGS.checkpoint) if ARGS.checkpoint else find_latest_checkpoint()
    if checkpoint is None:
        print("[Error] No checkpoint found. Pass --checkpoint or place policy_* outputs under this directory.")
        sys.exit(1)

    print(f"Using checkpoint: {checkpoint}")
    print(f"Visualization directory: {BASE_DIR}")
    print(f"Ray CPUs: {ARGS.num_cpus}, PyTorch threads: {ARGS.num_threads}")

    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, num_cpus=ARGS.num_cpus, log_to_driver=False)

    env = swimmer_gym({})
    obs = env.reset()

    agent = ppo.PPO(config=build_config(), env=swimmer_gym)
    try:
        agent.restore(str(checkpoint))
    except Exception as exc:
        print(f"[Error] Failed to restore checkpoint: {exc}")
        ray.shutdown()
        sys.exit(1)

    policy = agent.get_policy()
    state = policy.get_initial_state()
    centroid_trace = deque(maxlen=ARGS.trace_len)

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Flagella Self-Propel Policy Visualization")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    body_line, = ax.plot([], [], "-", lw=3, markersize=5, color="royalblue", label="Robot Body")
    trace_line, = ax.plot([], [], "-", lw=1.5, color="black", alpha=0.55, label="Centroid Trace")
    heading_line, = ax.plot([], [], "--", color="green", lw=2.0, label="Average Heading")
    head_line, = ax.plot([], [], "-", color="red", lw=2.0, label="First Link Heading")
    centroid_marker, = ax.plot([], [], "o", color="crimson", markersize=6, label="Centroid")
    info_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    plt.legend(loc="lower right")

    print("-" * 128)
    print(
        f"{'Step':<6} | {'Reward':<10} | {'P_rwd':<10} | {'Dir_pen':<10} | "
        f"{'Disp30':<10} | {'Gate30':<10} | {'Scale':<8} | {'HeadDeg':<10} | {'AvgDeg':<10}"
    )
    print("-" * 128)

    try:
        for step_idx in range(1, ARGS.steps + 1):
            action_output = agent.compute_single_action(observation=obs, state=state, explore=False)
            action, state = unpack_action_output(action_output, state)
            obs, reward, done, _ = env.step(action)

            body_points = env.XY_positions.copy()
            centroid_x = float(env.state[0])
            centroid_y = float(env.state[1])
            centroid_trace.append((centroid_x, centroid_y))

            first_link_heading = float(env.state[2])
            average_heading = compute_average_heading(env.state)
            first_link_deg = math.degrees(first_link_heading)
            average_deg = math.degrees(average_heading)

            body_line.set_data(body_points[:, 0], body_points[:, 1])
            centroid_marker.set_data([centroid_x], [centroid_y])

            if centroid_trace:
                trace_arr = np.array(centroid_trace)
                trace_line.set_data(trace_arr[:, 0], trace_arr[:, 1])

            line_len = 1.8
            heading_line.set_data(
                [centroid_x, centroid_x + line_len * math.cos(average_heading)],
                [centroid_y, centroid_y + line_len * math.sin(average_heading)],
            )
            head_line.set_data(
                [centroid_x, centroid_x + line_len * math.cos(first_link_heading)],
                [centroid_y, centroid_y + line_len * math.sin(first_link_heading)],
            )

            ax.set_xlim(centroid_x - ARGS.view_range, centroid_x + ARGS.view_range)
            ax.set_ylim(centroid_y - ARGS.view_range, centroid_y + ARGS.view_range)

            info_text.set_text(
                f"Step: {step_idx}\n"
                f"Reward: {reward:.4f}\n"
                f"P_rwd: {env.last_pressure_reward:.4f}\n"
                f"Dir_pen: {env.last_direction_penalty:.4f}\n"
                f"Disp30: {env.last_recent_displacement:.4f}\n"
                f"Gate30: {env.displacement_gate_ref:.4f}\n"
                f"Dir_scale: {env.last_displacement_scale:.3f}\n"
                f"HeadDeg: {first_link_deg:.2f}\n"
                f"AvgDeg: {average_deg:.2f}"
            )

            fig.canvas.draw()
            fig.canvas.flush_events()

            print(
                f"{step_idx:<6} | {reward:<10.4f} | {env.last_pressure_reward:<10.4f} | "
                f"{env.last_direction_penalty:<10.4f} | {env.last_recent_displacement:<10.4f} | "
                f"{env.displacement_gate_ref:<10.4f} | {env.last_displacement_scale:<8.3f} | "
                f"{first_link_deg:<10.2f} | {average_deg:<10.2f}"
            )

            if done:
                obs = env.reset()
                state = policy.get_initial_state()
                centroid_trace.clear()

            if not plt.fignum_exists(fig.number):
                break
            if ARGS.speed > 0:
                plt.pause(ARGS.speed)
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user.")
    finally:
        agent.stop()
        ray.shutdown()

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
