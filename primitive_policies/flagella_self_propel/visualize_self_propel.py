import argparse
import math
import os
import sys
from collections import deque
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a trained flagella self-propel policy")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file or containing directory. Defaults to latest local policy.")
    parser.add_argument("--steps", type=int, default=2000, help="Visualization steps to simulate.")
    parser.add_argument("--speed", type=float, default=0.03, help="Pause duration between frames in seconds.")
    parser.add_argument("--num_cpus", type=int, default=1, help="Number of CPUs for Ray during visualization.")
    parser.add_argument("--num_threads", type=int, default=5, help="Number of PyTorch threads for the Stokes solver.")
    parser.add_argument("--view_range", type=float, default=4.5, help="Half-width of the plotting window around the swimmer centroid.")
    parser.add_argument("--grid_spacing", type=float, default=0.6, help="Spacing of the fluid-visualization lattice.")
    parser.add_argument("--body_mask_radius", type=float, default=0.18, help="Hide fluid arrows too close to the swimmer body.")
    parser.add_argument("--flow_gain", type=float, default=0.18, help="Display gain applied to fluid velocity vectors before plotting.")
    parser.add_argument("--flow_clip", type=float, default=0.35, help="Maximum displayed arrow length after scaling.")
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
import torch

import calculate_v as cv
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


def compute_solver_diagnostics(state, action, x_first):
    L, e, Y, theta, action_absolute, Qu, Q1, Ql, Q2, _, beta_ini, absU, Xini, Yini = cv.initial(state, action, x_first)
    Y_dense = cv.initial_dense(state, action, x_first)

    cv.Xf_all_fila = Y[:, 0].clone().view(-1, 1, 1)
    cv.Zf_all_fila = Y[:, 1].clone().view(-1, 1, 1)

    for m in range(cv.Xf_match_q_fila.shape[1]):
        count = int(cv.Min_Distance_num_fila[m].item())
        selected_x = cv.Label_Matrix_fila[:, m] * Y_dense[:, 0]
        selected_z = cv.Label_Matrix_fila[:, m] * Y_dense[:, 1]
        selected_idx = np.nonzero(cv.Label_Matrix_fila[:, m])
        cv.Xf_match_q_fila[:, m, 0:count] = selected_x[selected_idx].view(1, -1)
        cv.Zf_match_q_fila[:, m, 0:count] = selected_z[selected_idx].view(1, -1)

    B = cv.MatrixB(L, theta, Y)
    A, Ap = cv.M1M2(e)
    B_supply = torch.zeros((3, A.shape[0] - B.shape[1]), dtype=torch.double, device=cv.device)
    B_all = torch.cat((B, B_supply), dim=1)

    Q = cv.MatrixQ(L, theta, Qu, Q1, Ql, Q2)
    C1, C2 = cv.MatrixC(action_absolute)
    AB = torch.linalg.solve(A.T, B_all.T).T.double()
    AB = AB[:, : B.shape[1]]
    MT = torch.matmul(AB, Q)
    M = torch.matmul(MT, C1)
    R = -torch.matmul(MT, C2)
    velo = torch.matmul(torch.linalg.inv(M), R)
    velo_points = torch.matmul(C1, velo) + C2
    velo_points_all = torch.matmul(Q, velo_points)

    velo_points_fila = torch.zeros((cv.Fila_point_num * 3, 1), dtype=torch.double, device=cv.device)
    velo_points_fila[: cv.Fila_point_num * 2, :] = velo_points_all
    force_points_fila = torch.linalg.solve(A, velo_points_fila)
    pressure_all = torch.matmul(Ap, force_points_fila.reshape(-1, 1))

    body_points = Y.detach().cpu().numpy()
    body_forces = force_points_fila.reshape(-1, 3).detach().cpu().numpy()
    pressure_np = pressure_all.view(-1).detach().cpu().numpy()
    return {
        "epsilon": float(e),
        "body_points": body_points,
        "body_forces": body_forces,
        "pressure_all": pressure_np,
    }


def compute_flow_vectors(query_points, source_points, source_forces, epsilon):
    dx = query_points[:, None, 0] - source_points[None, :, 0]
    dz = query_points[:, None, 1] - source_points[None, :, 1]
    r = np.sqrt(dx * dx + dz * dz + epsilon * epsilon)
    inv_r = 1.0 / r
    inv_r3 = inv_r**3

    s_xx = inv_r + epsilon * epsilon * inv_r3 + dx * dx * inv_r3
    s_xz = dx * dz * inv_r3
    s_zz = inv_r + epsilon * epsilon * inv_r3 + dz * dz * inv_r3

    fx = source_forces[None, :, 0]
    fz = source_forces[None, :, 1]

    u = np.sum(s_xx * fx + s_xz * fz, axis=1) / (8.0 * math.pi * cv.mu)
    v = np.sum(s_xz * fx + s_zz * fz, axis=1) / (8.0 * math.pi * cv.mu)
    return np.column_stack((u, v))


def point_to_polyline_distance(points, polyline):
    if len(polyline) < 2:
        return np.linalg.norm(points - polyline[0], axis=1)

    min_dist = np.full(points.shape[0], np.inf, dtype=np.float64)
    for start, end in zip(polyline[:-1], polyline[1:]):
        segment = end - start
        denom = float(np.dot(segment, segment))
        if denom < 1e-12:
            dist = np.linalg.norm(points - start, axis=1)
        else:
            rel = points - start
            t = np.clip(np.dot(rel, segment) / denom, 0.0, 1.0)
            proj = start + t[:, None] * segment
            dist = np.linalg.norm(points - proj, axis=1)
        min_dist = np.minimum(min_dist, dist)
    return min_dist


def build_flow_grid(center_x, center_y, spacing, view_range):
    xs = np.arange(center_x - view_range, center_x + view_range + 0.5 * spacing, spacing)
    ys = np.arange(center_y - view_range, center_y + view_range + 0.5 * spacing, spacing)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    return grid_x, grid_y, points


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
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_title("Flagella Self-Propel Policy Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    body_line, = ax.plot([], [], "-", lw=3.0, color="royalblue", marker="o", markersize=4, label="Swimmer")
    trace_line, = ax.plot([], [], "-", lw=1.2, color="black", alpha=0.55, label="Centroid trace")
    centroid_marker, = ax.plot([], [], "o", color="crimson", markersize=6, label="Centroid")
    grid_points_artist = None
    quiver = None
    info_text = ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )
    ax.legend(loc="lower right")

    print("-" * 118)
    print(
        f"{'Step':<6} | {'Reward':<10} | {'P_rwd':<10} | {'Dir_pen':<10} | "
        f"{'Disp30':<10} | {'Gate30':<10} | {'Scale':<8} | {'PressureDiffSum':<16}"
    )
    print("-" * 118)

    try:
        for step_idx in range(1, ARGS.steps + 1):
            action_output = agent.compute_single_action(observation=obs, state=state, explore=False)
            action, state = unpack_action_output(action_output, state)
            obs, reward, done, _ = env.step(action)

            diagnostics = compute_solver_diagnostics(env.state.copy(), action.copy(), env.Xfirst.copy())
            body_points = env.XY_positions.copy()
            centroid_x = float(env.state[0])
            centroid_y = float(env.state[1])
            centroid_trace.append((centroid_x, centroid_y))

            grid_x, grid_y, query_points = build_flow_grid(centroid_x, centroid_y, ARGS.grid_spacing, ARGS.view_range)
            flow = compute_flow_vectors(
                query_points=query_points,
                source_points=diagnostics["body_points"],
                source_forces=diagnostics["body_forces"][:, :2],
                epsilon=diagnostics["epsilon"],
            )

            mask_dist = point_to_polyline_distance(query_points, body_points) < ARGS.body_mask_radius
            flow_display = flow * ARGS.flow_gain
            flow_mag = np.linalg.norm(flow_display, axis=1)
            clip_scale = np.minimum(1.0, ARGS.flow_clip / np.maximum(flow_mag, 1e-12))
            flow_display = flow_display * clip_scale[:, None]
            flow_display[mask_dist] = 0.0

            body_line.set_data(body_points[:, 0], body_points[:, 1])
            centroid_marker.set_data([centroid_x], [centroid_y])
            if centroid_trace:
                trace_arr = np.array(centroid_trace)
                trace_line.set_data(trace_arr[:, 0], trace_arr[:, 1])

            if grid_points_artist is not None:
                grid_points_artist.remove()
            grid_points_artist = ax.scatter(
                query_points[:, 0],
                query_points[:, 1],
                s=5,
                color="lightgray",
                alpha=0.6,
                zorder=1,
            )

            if quiver is not None:
                quiver.remove()
            quiver = ax.quiver(
                query_points[:, 0],
                query_points[:, 1],
                flow_display[:, 0],
                flow_display[:, 1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color="darkorange",
                width=0.0035,
                alpha=0.85,
                zorder=2,
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
                f"Grid spacing: {ARGS.grid_spacing:.2f}\n"
                f"Flow gain/clip: {ARGS.flow_gain:.2f}/{ARGS.flow_clip:.2f}"
            )

            fig.canvas.draw()
            fig.canvas.flush_events()

            print(
                f"{step_idx:<6} | {reward:<10.4f} | {env.last_pressure_reward:<10.4f} | "
                f"{env.last_direction_penalty:<10.4f} | {env.last_recent_displacement:<10.4f} | "
                f"{env.displacement_gate_ref:<10.4f} | {env.last_displacement_scale:<8.3f} | "
                f"{env.pressure_diff:<16.4f}"
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
