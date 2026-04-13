import argparse
import os
import sys
import traceback
from pathlib import Path
from queue import Full, Queue
from threading import Event, Thread


BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Triple-Flagella Senior Policy Visualizer")
    parser.add_argument("--forward_ckpt", type=str, required=True, help="Checkpoint path for the forward primitive policy")
    parser.add_argument("--cw_ckpt", type=str, required=True, help="Checkpoint path for the clockwise turn primitive policy")
    parser.add_argument("--ccw_ckpt", type=str, required=True, help="Checkpoint path for the counter-clockwise turn primitive policy")
    parser.add_argument("--checkpoint", type=str, default=None, help="Senior-policy checkpoint path")
    parser.add_argument("--steps", type=int, default=200, help="Total macro steps to visualize (default: 200)")
    parser.add_argument("--speed", type=float, default=0.01, help="Refresh interval per displayed frame in seconds (default: 0.01)")
    parser.add_argument("--num_cpus", type=int, default=1, help="Number of CPUs used by Ray (default: 1)")
    parser.add_argument("--num_threads", type=int, default=1, help="Number of PyTorch threads used by the solver (default: 1)")
    parser.add_argument("--view_range", type=float, default=5.0, help="Half-width of the camera-follow window (default: 5.0)")
    parser.add_argument(
        "--prefetch_queue_size",
        type=int,
        default=10,
        help="Maximum number of precomputed macro-step packages kept in the queue (default: 10)",
    )
    parser.add_argument(
        "--prefetch_warmup_steps",
        type=int,
        default=10,
        help="Number of macro-step packages to preload before playback starts (default: 10)",
    )
    return parser.parse_args()


ARGS = parse_args()
os.environ["STOKES_NUM_THREADS"] = str(ARGS.num_threads)

os.chdir(BASE_DIR)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import matplotlib

if sys.platform == "darwin":
    matplotlib.use("MacOSX")
else:
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import ray
import ray.rllib.algorithms.ppo as ppo

from swimmer import LOW_LEVEL_HOLD_STEPS, MACRO_ACTION_TABLE, MACRO_HORIZON, compute_average_heading, compute_true_centroid, swimmer_gym


def build_env_config(cli_args):
    return {
        "forward_ckpt": cli_args.forward_ckpt,
        "cw_ckpt": cli_args.cw_ckpt,
        "ccw_ckpt": cli_args.ccw_ckpt,
        "low_level_hold_steps": LOW_LEVEL_HOLD_STEPS,
        "macro_horizon": MACRO_HORIZON,
    }


def is_checkpoint_path(path_obj):
    path_obj = Path(path_obj)
    if not path_obj.exists():
        return False
    if path_obj.is_file():
        return path_obj.name.startswith("checkpoint-")
    return path_obj.name.startswith("checkpoint_") or (path_obj / "rllib_checkpoint.json").exists() or (path_obj / ".is_checkpoint").exists()


def checkpoint_sort_key(path_obj):
    path_obj = Path(path_obj)
    digits = "".join(ch for ch in path_obj.name if ch.isdigit())
    order = int(digits) if digits else -1
    return (order, str(path_obj))


def find_latest_checkpoint(base_dir=None):
    base_dir = Path(base_dir or BASE_DIR)
    policy_roots = [item for item in base_dir.iterdir() if item.is_dir() and item.name.startswith("policy_")]
    if not policy_roots:
        return None
    latest_policy = max(policy_roots, key=lambda item: item.stat().st_mtime)
    iter_dirs = [item for item in latest_policy.iterdir() if item.is_dir() and item.name.isdigit()]
    if not iter_dirs:
        return None
    latest_iter = max(iter_dirs, key=lambda item: int(item.name))
    candidates = sorted([item for item in latest_iter.rglob("*") if is_checkpoint_path(item)], key=checkpoint_sort_key)
    return candidates[-1] if candidates else None


def resolve_checkpoint(path_str):
    cp_path = Path(path_str).expanduser().resolve()
    if cp_path.is_file():
        return cp_path
    if not cp_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {cp_path}")
    if is_checkpoint_path(cp_path):
        return cp_path

    direct_candidates = sorted([candidate for candidate in cp_path.iterdir() if is_checkpoint_path(candidate)], key=checkpoint_sort_key)
    if direct_candidates:
        return direct_candidates[-1]

    nested_candidates = sorted([candidate for candidate in cp_path.rglob("*") if is_checkpoint_path(candidate)], key=checkpoint_sort_key)
    if nested_candidates:
        return nested_candidates[-1]

    raise FileNotFoundError(f"No checkpoint found under: {cp_path}")


def build_config():
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 0
    config["num_rollout_workers"] = 0
    config["framework"] = "torch"
    config["env_config"] = build_env_config(ARGS)
    config["gamma"] = 0.995
    config["lr"] = 0.0003
    config["horizon"] = MACRO_HORIZON
    config["rollout_fragment_length"] = MACRO_HORIZON
    config["evaluation_duration"] = 10000000
    config["lr_schedule"] = None
    config["use_critic"] = True
    config["use_gae"] = True
    config["lambda_"] = 0.95
    config["kl_coeff"] = 0.2
    config["sgd_minibatch_size"] = 100
    config["train_batch_size"] = 800
    config["num_sgd_iter"] = 10
    config["shuffle_sequences"] = True
    config["vf_loss_coeff"] = 1.0
    config["entropy_coeff"] = 0.001
    config["entropy_coeff_schedule"] = None
    config["clip_param"] = 0.1
    config["vf_clip_param"] = 100000
    config["grad_clip"] = None
    config["kl_target"] = 0.01
    config["evaluation_interval"] = 1000000
    config["evaluation_duration"] = 1
    config["use_lstm"] = False
    config["min_sample_timesteps_per_iteration"] = 500
    config["env"] = swimmer_gym
    return config


def unpack_action_output(action_output):
    if not isinstance(action_output, tuple):
        return action_output
    if len(action_output) == 0:
        raise ValueError("compute_single_action returned an empty tuple")
    return action_output[0]


def draw_heading(ax, centroid, heading_angle, color):
    length = 0.6
    dx = length * np.cos(heading_angle)
    dy = length * np.sin(heading_angle)
    ax.plot([centroid[0], centroid[0] + dx], [centroid[1], centroid[1] + dy], color=color, linewidth=2.0)


def capture_env_frame(env, substep_index):
    centroid1 = compute_true_centroid(env.XY_positions[0])
    centroid2 = compute_true_centroid(env.XY_positions[1])
    centroid3 = compute_true_centroid(env.XY_positions[2])
    return {
        "substep_index": int(substep_index),
        "xy1": np.array(env.XY_positions[0], copy=True),
        "xy2": np.array(env.XY_positions[1], copy=True),
        "xy3": np.array(env.XY_positions[2], copy=True),
        "state1": np.array(env.states[0], copy=True),
        "state2": np.array(env.states[1], copy=True),
        "state3": np.array(env.states[2], copy=True),
        "centroid1": np.array(centroid1, copy=True),
        "centroid2": np.array(centroid2, copy=True),
        "centroid3": np.array(centroid3, copy=True),
    }


def render_frame(
    ax,
    frame,
    traces,
    macro_index,
    action,
    reward,
    primitive_triple,
    total_substeps,
    forward_reward,
    dx12_penalty,
    dx23_penalty,
    dy12_penalty,
    dy23_penalty,
    dx12,
    dx23,
    dy12,
    dy23,
    queue_fill,
    queue_capacity,
):
    centroids = [np.array(frame["centroid1"], copy=True), np.array(frame["centroid2"], copy=True), np.array(frame["centroid3"], copy=True)]
    headings = [
        compute_average_heading(frame["state1"]),
        compute_average_heading(frame["state2"]),
        compute_average_heading(frame["state3"]),
    ]
    colors = ["tab:blue", "tab:red", "tab:orange"]
    xy_keys = ["xy1", "xy2", "xy3"]

    for idx in range(3):
        traces[idx].append(centroids[idx])

    ax.clear()
    for idx in range(3):
        ax.plot(frame[xy_keys[idx]][:, 0], frame[xy_keys[idx]][:, 1], color=colors[idx], linewidth=2.5)
        ax.scatter([centroids[idx][0]], [centroids[idx][1]], color=colors[idx], s=40)
        if len(traces[idx]) > 1:
            trace_np = np.array(traces[idx])
            ax.plot(trace_np[:, 0], trace_np[:, 1], color=colors[idx], alpha=0.5, linewidth=1.0)
        draw_heading(ax, centroids[idx], headings[idx], "tab:green")

    center_x = float(np.mean([centroid[0] for centroid in centroids]))
    center_y = float(np.mean([centroid[1] for centroid in centroids]))
    ax.set_xlim(center_x - ARGS.view_range, center_x + ARGS.view_range)
    ax.set_ylim(center_y - ARGS.view_range, center_y + ARGS.view_range)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)

    macro_triple = MACRO_ACTION_TABLE[action]
    substep_index = int(frame.get("substep_index", total_substeps))
    info_text = "\n".join(
        [
            f"Macro step: {macro_index}",
            f"Substep: {substep_index}/{total_substeps}",
            f"Macro action: {action} -> {macro_triple[0]} / {macro_triple[1]} / {macro_triple[2]}",
            f"Current primitive: {primitive_triple[0]} / {primitive_triple[1]} / {primitive_triple[2]}",
            f"Reward: {reward:.4f}",
            f"Forward: {forward_reward:.4f}",
            f"Dx12 penalty: {dx12_penalty:.4f}",
            f"Dx23 penalty: {dx23_penalty:.4f}",
            f"Dy12 penalty: {dy12_penalty:.4f}",
            f"Dy23 penalty: {dy23_penalty:.4f}",
            f"dx12: {dx12:.4f}, dx23: {dx23:.4f}",
            f"dy12: {dy12:.4f}, dy23: {dy23:.4f}",
            f"Buffer: {queue_fill}/{queue_capacity}",
        ]
    )
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.set_title("Triple Flagella Senior Policy Visualization")


def compute_macro_package(agent, env, obs):
    try:
        action_output = agent.compute_single_action(observation=obs, explore=False)
    except TypeError:
        action_output = agent.compute_single_action(obs, explore=False)
    action = int(unpack_action_output(action_output))
    next_obs, reward, done, _ = env.step(action)
    frames = env.last_substep_frames if env.last_substep_frames else [capture_env_frame(env, env.low_level_hold_steps)]
    return {
        "action": action,
        "reward": reward,
        "done": done,
        "next_obs": next_obs,
        "frames": frames,
        "macro_triple": MACRO_ACTION_TABLE[action],
        "forward_reward": env.last_forward_reward,
        "dx12_penalty": env.last_dx12_penalty,
        "dx23_penalty": env.last_dx23_penalty,
        "dy12_penalty": env.last_dy12_penalty,
        "dy23_penalty": env.last_dy23_penalty,
        "dx12": env.last_dx12,
        "dx23": env.last_dx23,
        "dy12": env.last_dy12,
        "dy23": env.last_dy23,
    }


def producer_loop(agent, env, initial_obs, output_queue, stop_event, total_steps):
    obs = initial_obs
    produced = 0
    try:
        while produced < total_steps and not stop_event.is_set():
            package = compute_macro_package(agent, env, obs)
            while not stop_event.is_set():
                try:
                    output_queue.put(package, timeout=0.1)
                    break
                except Full:
                    continue
            if stop_event.is_set():
                break
            produced += 1
            obs = package["next_obs"]
            if package["done"]:
                obs = env.reset()
    except Exception:
        error_package = {"error": True, "traceback": traceback.format_exc()}
        while not stop_event.is_set():
            try:
                output_queue.put(error_package, timeout=0.1)
                break
            except Full:
                continue
    finally:
        sentinel = {"done_producing": True}
        while not stop_event.is_set():
            try:
                output_queue.put(sentinel, timeout=0.1)
                break
            except Full:
                continue


def main():
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, num_cpus=ARGS.num_cpus, log_to_driver=False)

    env = swimmer_gym(build_env_config(ARGS))
    obs = env.reset()

    agent = ppo.PPO(config=build_config(), env=swimmer_gym)
    checkpoint = resolve_checkpoint(ARGS.checkpoint) if ARGS.checkpoint else find_latest_checkpoint()
    if checkpoint is None:
        print("[Error] No senior-policy checkpoint found. Run train.py first or pass --checkpoint.")
        sys.exit(1)

    print(f"Loading senior checkpoint: {checkpoint}")
    print(f"Ray CPUs: {ARGS.num_cpus}, PyTorch threads: {ARGS.num_threads}")
    agent.restore(str(checkpoint))
    print(">>> Checkpoint restore succeeded. Launching visualization window...")

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    traces = [[], [], []]
    initial_frame = capture_env_frame(env, substep_index=0)
    render_frame(
        ax,
        initial_frame,
        traces,
        macro_index=0,
        action=0,
        reward=0.0,
        primitive_triple=("forward", "forward", "forward"),
        total_substeps=1,
        forward_reward=0.0,
        dx12_penalty=0.0,
        dx23_penalty=0.0,
        dy12_penalty=0.0,
        dy23_penalty=0.0,
        dx12=0.0,
        dx23=0.0,
        dy12=0.0,
        dy23=0.0,
        queue_fill=0,
        queue_capacity=max(1, ARGS.prefetch_queue_size),
    )
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(ARGS.speed)

    preload_size = max(1, ARGS.prefetch_queue_size)
    package_queue = Queue(maxsize=preload_size)
    stop_event = Event()
    producer = Thread(target=producer_loop, args=(agent, env, obs, package_queue, stop_event, ARGS.steps), daemon=True)
    producer.start()
    macro_index = 0

    warmup_steps = min(max(1, ARGS.prefetch_warmup_steps), preload_size, ARGS.steps)
    print(f">>> Warming up playback queue: target {warmup_steps} macro steps (queue size {preload_size})")
    while package_queue.qsize() < warmup_steps and producer.is_alive():
        if not plt.fignum_exists(fig.number):
            stop_event.set()
            producer.join(timeout=2.0)
            plt.ioff()
            return
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.05)

    if package_queue.qsize() > 0:
        print(f">>> Playback queue warmup complete: {package_queue.qsize()} macro steps buffered")

    try:
        while macro_index < ARGS.steps:
            package = package_queue.get()
            if package.get("error"):
                raise RuntimeError(package["traceback"])
            if package.get("done_producing"):
                break

            macro_index += 1
            action = package["action"]
            reward = package["reward"]
            macro_triple = package["macro_triple"]

            print(
                f"[Macro {macro_index:>3d}] action={action} "
                f"({macro_triple[0]}-{macro_triple[1]}-{macro_triple[2]}) | "
                f"reward={reward:.4f} | forward={package['forward_reward']:.4f} | "
                f"dx12_pen={package['dx12_penalty']:.4f} | dx23_pen={package['dx23_penalty']:.4f} | "
                f"dy12_pen={package['dy12_penalty']:.4f} | dy23_pen={package['dy23_penalty']:.4f} | "
                f"dx12={package['dx12']:.4f} dx23={package['dx23']:.4f} "
                f"dy12={package['dy12']:.4f} dy23={package['dy23']:.4f}"
            )

            should_stop = False
            for frame in package["frames"]:
                render_frame(
                    ax,
                    frame,
                    traces,
                    macro_index=macro_index,
                    action=action,
                    reward=reward,
                    primitive_triple=macro_triple,
                    total_substeps=len(package["frames"]),
                    forward_reward=package["forward_reward"],
                    dx12_penalty=package["dx12_penalty"],
                    dx23_penalty=package["dx23_penalty"],
                    dy12_penalty=package["dy12_penalty"],
                    dy23_penalty=package["dy23_penalty"],
                    dx12=package["dx12"],
                    dx23=package["dx23"],
                    dy12=package["dy12"],
                    dy23=package["dy23"],
                    queue_fill=package_queue.qsize(),
                    queue_capacity=preload_size,
                )
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(ARGS.speed)

                if not plt.fignum_exists(fig.number):
                    should_stop = True
                    break

            if should_stop:
                break
    finally:
        stop_event.set()
        producer.join(timeout=2.0)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
