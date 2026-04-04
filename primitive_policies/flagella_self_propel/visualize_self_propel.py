import argparse
import math
import os
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Self-Propel Task Real-Time Visualizer")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. If omitted, automatically use the latest local policy.",
    )
    parser.add_argument("--steps", type=int, default=2000, help="Simulation steps to visualize.")
    parser.add_argument("--speed", type=float, default=0.03, help="Refresh interval in seconds.")
    parser.add_argument("--num_cpus", type=int, default=1, help="Number of CPUs for Ray.")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=5,
        help="Number of PyTorch threads for the Stokes solver.",
    )
    parser.add_argument(
        "--view_range",
        type=float,
        default=4.5,
        help="Half-width of the plotting window around the swimmer centroid.",
    )
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
    if not base_dir.exists():
        return None, None

    policy_roots = [path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith("policy_")]
    if not policy_roots:
        return None, None

    latest_policy = max(policy_roots, key=lambda path: path.stat().st_mtime)
    iter_dirs = [path for path in latest_policy.iterdir() if path.is_dir() and path.name.isdigit()]
    if not iter_dirs:
        return None, latest_policy

    latest_iter_dir = max(iter_dirs, key=lambda path: int(path.name))
    checkpoint_paths = sorted(
        [candidate for candidate in latest_iter_dir.rglob("*") if is_checkpoint_path(candidate)],
        key=checkpoint_sort_key,
    )
    if not checkpoint_paths:
        return None, latest_policy

    return checkpoint_paths[-1], latest_policy


def resolve_checkpoint(path_str):
    checkpoint_path = Path(path_str).expanduser().resolve()
    if checkpoint_path.is_file():
        return checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    if is_checkpoint_path(checkpoint_path):
        return checkpoint_path

    direct_candidates = sorted(
        [candidate for candidate in checkpoint_path.iterdir() if is_checkpoint_path(candidate)],
        key=checkpoint_sort_key,
    )
    if direct_candidates:
        return direct_candidates[-1]

    nested_candidates = sorted(
        [candidate for candidate in checkpoint_path.rglob("*") if is_checkpoint_path(candidate)],
        key=checkpoint_sort_key,
    )
    if nested_candidates:
        return nested_candidates[-1]

    raise FileNotFoundError(f"No checkpoint found under: {checkpoint_path}")


def get_config():
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
    heading = float(state[2])
    running_angle = heading
    angle_sum = heading
    for beta in state[3:]:
        running_angle += float(beta)
        angle_sum += running_angle
    return angle_sum / (len(state) - 2)


def main():
    args = ARGS

    checkpoint_path = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    if checkpoint_path is None:
        print("Searching for the latest local policy...")
        checkpoint_path, policy_root = find_latest_checkpoint()
        if checkpoint_path is None:
            print("[Error] No checkpoint found. Run train.py first or pass --checkpoint.")
            sys.exit(1)
        print(f"Found latest policy under:\n  {policy_root}")
    else:
        checkpoint_path = resolve_checkpoint(str(checkpoint_path))

    print("=" * 48)
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Visualization directory: {BASE_DIR}")
    print(f"Ray CPUs: {args.num_cpus}, PyTorch threads: {args.num_threads}")
    print("=" * 48)

    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, num_cpus=args.num_cpus, log_to_driver=False)

    env = swimmer_gym({})
    obs = env.reset()

    origin_x = env.X_ini
    origin_y = env.Y_ini

    config = get_config()
    agent = ppo.PPO(config=config, env=swimmer_gym)

    print("Loading checkpoint...")
    try:
        agent.restore(str(checkpoint_path))
        print(">>> Checkpoint restore succeeded. Starting visualization...")
    except Exception as error:
        print(f"[Error] Failed to restore checkpoint: {error}")
        ray.shutdown()
        sys.exit(1)

    policy = agent.get_policy()
    state = policy.get_initial_state()

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Self-Propel Task Visualization")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    body_line, = ax.plot([], [], "-", lw=3, markersize=5, color="royalblue", label="Robot Body")
    average_heading_line, = ax.plot([], [], "--", color="green", lw=2.0, label="Average Heading")
    first_link_line, = ax.plot([], [], "-", color="red", lw=2.0, label="First Link Heading")
    centroid_marker, = ax.plot([], [], "o", color="crimson", markersize=6, label="Centroid")
    ax.plot(origin_x, origin_y, "kx", markersize=8, label="Initial Centroid")
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

    print("-" * 118)
    print(
        f"{'Step':<8} | {'AvgDeg':<10} | {'HeadDeg':<10} | {'Reward':<10} | "
        f"{'P_rwd':<10} | {'Dir_pen':<10} | {'Disp30':<10} | {'Scale':<8}"
    )
    print("-" * 118)

    try:
        for step_idx in range(args.steps):
            action_output = agent.compute_single_action(observation=obs, state=state, explore=False)
            action, state = unpack_action_output(action_output, state)
            obs, reward, done, _ = env.step(action)

            robot_shape = env.XY_positions.copy()
            current_x = robot_shape[:, 0]
            current_y = robot_shape[:, 1]

            centroid_x = float(env.state[0])
            centroid_y = float(env.state[1])
            first_link_heading = float(env.state[2])
            average_heading = compute_average_heading(env.state)
            first_link_deg = math.degrees(first_link_heading)
            average_deg = math.degrees(average_heading)

            body_line.set_data(current_x, current_y)
            centroid_marker.set_data([centroid_x], [centroid_y])

            line_len = 2.5
            average_heading_line.set_data(
                [centroid_x, centroid_x + line_len * math.cos(average_heading)],
                [centroid_y, centroid_y + line_len * math.sin(average_heading)],
            )
            first_link_line.set_data(
                [centroid_x, centroid_x + line_len * math.cos(first_link_heading)],
                [centroid_y, centroid_y + line_len * math.sin(first_link_heading)],
            )

            ax.set_xlim(centroid_x - args.view_range, centroid_x + args.view_range)
            ax.set_ylim(centroid_y - args.view_range, centroid_y + args.view_range)

            info_text.set_text(
                f"Step: {step_idx + 1}\n"
                f"Reward: {reward:.4f}\n"
                f"P_rwd: {env.last_pressure_reward:.4f}\n"
                f"Dir_pen: {env.last_direction_penalty:.4f}\n"
                f"Disp30: {env.last_recent_displacement:.4f}\n"
                f"Gate30: {env.displacement_gate_ref:.4f}\n"
                f"Dir_scale: {env.last_displacement_scale:.3f}\n"
                f"AvgDeg: {average_deg:.2f}\n"
                f"HeadDeg: {first_link_deg:.2f}"
            )

            fig.canvas.draw()
            fig.canvas.flush_events()

            print(
                f"{step_idx + 1:<8} | {average_deg:<10.2f} | {first_link_deg:<10.2f} | {reward:<10.4f} | "
                f"{env.last_pressure_reward:<10.4f} | {env.last_direction_penalty:<10.4f} | "
                f"{env.last_recent_displacement:<10.4f} | {env.last_displacement_scale:<8.3f}"
            )

            if not plt.fignum_exists(fig.number):
                break

            if done:
                obs = env.reset()
                state = policy.get_initial_state()

            if args.speed > 0:
                plt.pause(args.speed)
    except KeyboardInterrupt:
        print("\nVisualization interrupted.")
    finally:
        agent.stop()
        ray.shutdown()

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
