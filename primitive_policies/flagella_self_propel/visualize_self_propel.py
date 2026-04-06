import argparse
import os
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


# ================= 3. 参数解析 =================
def parse_args():
    parser = argparse.ArgumentParser(description="Micro-Robot Real-Time Visualizer")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. If omitted, auto-detect the latest checkpoint under local policy_* folders.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Total visualization steps (default: 2000)",
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=0.001,
        help="Refresh interval in seconds; smaller is faster (default: 0.001)",
    )

    parser.add_argument(
        "--num_cpus",
        type=int,
        default=1,
        help="Number of CPUs used by Ray (default: 1)",
    )

    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of PyTorch threads used by the Stokes solver (default: 1)",
    )

    parser.add_argument(
        "--view_range",
        type=float,
        default=4.0,
        help="Half-width of the camera-follow window (default: 4.0)",
    )

    return parser.parse_args()


ARGS = parse_args()
os.environ["STOKES_NUM_THREADS"] = str(ARGS.num_threads)


# ================= 1. 强制设置绘图后端 (关键) =================
os.chdir(BASE_DIR)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import matplotlib

# macOS 上优先使用原生后端，减少空白窗口和无响应问题
if sys.platform == "darwin":
    matplotlib.use("MacOSX")
else:
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
        return None

    policy_roots = [path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith("policy_")]
    if not policy_roots:
        return None

    latest_policy = max(policy_roots, key=lambda path: path.stat().st_mtime)
    iter_dirs = [path for path in latest_policy.iterdir() if path.is_dir() and path.name.isdigit()]
    if not iter_dirs:
        return None

    latest_iter_dir = max(iter_dirs, key=lambda path: int(path.name))
    checkpoint_paths = sorted(
        [candidate for candidate in latest_iter_dir.rglob("*") if is_checkpoint_path(candidate)],
        key=checkpoint_sort_key,
    )
    if not checkpoint_paths:
        return None

    return checkpoint_paths[-1]


def resolve_checkpoint(path_str):
    cp_path = Path(path_str).expanduser().resolve()
    if cp_path.is_file():
        return cp_path
    if not cp_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {cp_path}")

    if is_checkpoint_path(cp_path):
        return cp_path

    direct_candidates = sorted(
        [candidate for candidate in cp_path.iterdir() if is_checkpoint_path(candidate)],
        key=checkpoint_sort_key,
    )
    if direct_candidates:
        return direct_candidates[-1]

    nested_candidates = sorted(
        [candidate for candidate in cp_path.rglob("*") if is_checkpoint_path(candidate)],
        key=checkpoint_sort_key,
    )
    if nested_candidates:
        return nested_candidates[-1]

    raise FileNotFoundError(f"No checkpoint found under: {cp_path}")


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


def compute_average_heading(state_array):
    head_omega = state_array[2]
    running_angle = head_omega
    angle_sum = head_omega
    for beta in state_array[3:]:
        running_angle += beta
        angle_sum += running_angle
    return angle_sum / (len(state_array) - 2)


def compute_true_centroid(robot_shape):
    return np.mean(robot_shape[:, 0]), np.mean(robot_shape[:, 1])


# ================= 2. 配置函数 (需与 train.py 一致) =================
def get_config():
    """
    复制 train.py 中的关键配置，确保模型能正确加载。
    """
    config = ppo.DEFAULT_CONFIG.copy()

    # 资源与框架
    config["num_gpus"] = 0
    config["num_workers"] = 0
    config["num_rollout_workers"] = 0
    config["framework"] = "torch"

    # 网络结构 (LSTM)
    config["use_lstm"] = True
    config["max_seq_len"] = 100

    # 环境与训练关键参数
    config["batch_mode"] = "complete_episodes"
    config["rollout_fragment_length"] = 3000
    config["horizon"] = 3000
    config["gamma"] = 0.9999
    config["lr"] = 0.0003
    config["evaluation_duration"] = 10000000
    config["lr_schedule"] = None
    config["use_critic"] = True
    config["use_gae"] = True
    config["lambda_"] = 0.98
    config["kl_coeff"] = 0.2
    config["sgd_minibatch_size"] = 256
    config["train_batch_size"] = 6000
    config["num_sgd_iter"] = 15
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
    config["min_sample_timesteps_per_iteration"] = 6000

    # 必须传入环境类
    config["env"] = swimmer_gym

    return config


# ================= 4. 主程序 =================
def main():
    args = ARGS

    # --- 初始化 Ray ---
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, num_cpus=args.num_cpus, log_to_driver=False)

    # --- 实例化环境和 Agent ---
    env = swimmer_gym({})
    obs = env.reset()

    config = get_config()
    agent = ppo.PPO(config=config, env=swimmer_gym)

    # --- 加载模型 (路径自动纠错 + 自动寻找最新 checkpoint) ---
    if args.checkpoint:
        cp_path = resolve_checkpoint(args.checkpoint)
    else:
        cp_path = find_latest_checkpoint()
        if cp_path is None:
            print("\n[Error] No checkpoint found. Run train.py first or pass --checkpoint.")
            sys.exit(1)

    print(f"Loading checkpoint: {cp_path}")
    print(f"Ray CPUs: {args.num_cpus}, PyTorch threads: {args.num_threads}")
    try:
        agent.restore(str(cp_path))
        print(">>> Checkpoint restore succeeded. Launching visualization window...")
    except Exception as e:
        print(f"\n[Error] Failed to restore checkpoint: {e}")
        print("Please check the path, checkpoint layout, or RLlib version.")
        sys.exit(1)

    # --- 准备绘图窗口 (实时模式) ---
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Real-Time Simulation (Running...)")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    # 初始化绘图元素
    # 机器人身体 (蓝色点线)
    line, = ax.plot([], [], "-", lw=2, markersize=4, color="royalblue", label="Robot")
    # 质心轨迹 (红色细线)
    trace, = ax.plot([], [], "-", lw=1, color="crimson", alpha=0.5, label="Trace")
    # 平均朝向辅助线 (绿色虚线)
    avg_line, = ax.plot([], [], "--", lw=2, color="green", alpha=0.8, label="Average Heading")
    # 质心点
    centroid_dot, = ax.plot([], [], "o", color="black", markersize=5, label="Centroid")
    # 文字信息
    info_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        va="top",
    )

    plt.legend(loc="upper right")
    plt.show(block=False)

    # --- 初始化 LSTM 状态 ---
    policy = agent.get_policy()
    state = policy.get_initial_state()

    # 数据容器
    history_x = []
    history_y = []
    total_reward = 0.0

    print("-" * 110)
    print(
        f"{'Step':<10} | {'X Coord':<12} | {'Y Coord':<12} | {'Reward':<12} | "
        f"{'P_rwd':<10} | {'DirPenTot':<10} | {'Drift400':<10} | {'Disp100':<10}"
    )
    print("-" * 110)

    # 先显示初始几何，避免首帧空白。
    robot_shape = env.XY_positions.copy()
    current_x = robot_shape[:, 0]
    current_y = robot_shape[:, 1]
    centroid_x, centroid_y = compute_true_centroid(robot_shape)
    average_heading = compute_average_heading(env.state)
    history_x.append(centroid_x)
    history_y.append(centroid_y)
    line.set_data(current_x, current_y)
    trace.set_data(history_x, history_y)
    centroid_dot.set_data([centroid_x], [centroid_y])
    line_len = 2.0
    avg_line.set_data(
        [centroid_x, centroid_x + line_len * np.cos(average_heading)],
        [centroid_y, centroid_y + line_len * np.sin(average_heading)],
    )
    ax.set_xlim(centroid_x - args.view_range, centroid_x + args.view_range)
    ax.set_ylim(centroid_y - args.view_range, centroid_y + args.view_range)
    info_text.set_text(
        f"Step: 0\n"
        f"X: {centroid_x:.2f}\n"
        f"Y: {centroid_y:.2f}\n"
        f"Reward: 0.00\n"
        f"P_rwd: 0.000\n"
        f"LocalDirPen: 0.000\n"
        f"DriftBiasPen: 0.000\n"
        f"DirPenTotal: 0.000\n"
        f"CumDriftDeg400: 0.00\n"
        f"Disp100: 0.000\n"
        f"Gate100: {env.displacement_gate_ref:.3f}"
    )
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.05)

    # ================= 5. 实时模拟循环 =================
    try:
        for i in range(args.steps):
            if not plt.fignum_exists(fig.number):
                print("\nWindow closed. Stop simulation.")
                break

            # 先处理一次 GUI 事件，避免窗口长时间无响应。
            plt.pause(0.001)

            # (1) 计算动作
            action_output = agent.compute_single_action(
                observation=obs,
                state=state,
                explore=False,
            )
            action, state = unpack_action_output(action_output, state)

            # (2) 环境步进
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # (3) 获取数据
            # env.XY_positions is an (N+1, 2) array
            robot_shape = env.XY_positions.copy()
            current_x = robot_shape[:, 0]
            current_y = robot_shape[:, 1]

            centroid_x, centroid_y = compute_true_centroid(robot_shape)
            average_heading = compute_average_heading(env.state)

            # 记录轨迹 (取质心)
            history_x.append(centroid_x)
            history_y.append(centroid_y)

            # (4) 更新画面
            # 更新机器人身体
            line.set_data(current_x, current_y)

            # 更新轨迹 (为保持流畅，只画最近 1000 步)
            trace_len = 1000
            if len(history_x) > trace_len:
                trace.set_data(history_x[-trace_len:], history_y[-trace_len:])
            else:
                trace.set_data(history_x, history_y)

            centroid_dot.set_data([centroid_x], [centroid_y])

            line_len = 2.0
            avg_line.set_data(
                [centroid_x, centroid_x + line_len * np.cos(average_heading)],
                [centroid_y, centroid_y + line_len * np.sin(average_heading)],
            )
            # 动态调整相机视野 (Camera Follow)
            center_x = np.mean(current_x)
            center_y = np.mean(current_y)
            view_range = args.view_range
            ax.set_xlim(center_x - view_range, center_x + view_range)
            ax.set_ylim(center_y - view_range, center_y + view_range)

            # 更新文字
            info_text.set_text(
                f"Step: {i + 1}\n"
                f"X: {centroid_x:.2f}\n"
                f"Y: {centroid_y:.2f}\n"
                f"Reward: {total_reward:.2f}\n"
                f"P_rwd: {env.last_pressure_reward:.3f}\n"
                f"LocalDirPen: {env.last_direction_penalty:.3f}\n"
                f"DriftBiasPen: {env.last_drift_bias_penalty:.3f}\n"
                f"DirPenTotal: {env.last_total_direction_penalty:.3f}\n"
                f"CumDriftDeg400: {env.last_cumulative_drift_deg:.2f}\n"
                f"Disp100: {env.last_recent_displacement:.3f}\n"
                f"Gate100: {env.displacement_gate_ref:.3f}"
            )

            # 刷新画布
            fig.canvas.draw()
            fig.canvas.flush_events()

            # 终端打印
            print(
                f"{i + 1:<10} | {centroid_x:<12.4f} | {centroid_y:<12.4f} | {total_reward:<12.4f} | "
                f"{env.last_pressure_reward:<10.4f} | {env.last_direction_penalty:<10.4f} | "
                f"{env.last_recent_displacement:<10.4f}"
            )

            # 检查窗口是否被用户关闭
            if done:
                obs = env.reset()
                state = policy.get_initial_state()

            # 稍微暂停以控制速度
            if args.speed > 0:
                plt.pause(args.speed)

    except KeyboardInterrupt:
        print("\nVisualization interrupted by user.")

    print("-" * 110)
    print("Simulation finished. Close the window to exit.")

    plt.ioff()
    plt.show()

    agent.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
