import argparse
import os
import shutil
from datetime import datetime
from pprint import pformat

import os.path as osp

parser = argparse.ArgumentParser(description="Train dual-flagella shared-policy senior policy")
parser.add_argument("--forward_ckpt", type=str, required=True, help="Checkpoint path for the forward primitive policy")
parser.add_argument("--cw_ckpt", type=str, required=True, help="Checkpoint path for the clockwise turn primitive policy")
parser.add_argument("--ccw_ckpt", type=str, required=True, help="Checkpoint path for the counter-clockwise turn primitive policy")
parser.add_argument("--num_cpus", type=int, default=5, help="Number of CPUs for Ray (default: 5)")
parser.add_argument("--num_threads", type=int, default=5, help="Number of PyTorch threads (default: 5)")
args = parser.parse_args()
os.environ["STOKES_NUM_THREADS"] = str(args.num_threads)

import ray
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.logger import pretty_print

import swimmer as swimmer_module
from swimmer import PRIMITIVE_NAMES, ROBOT_IDS, swimmer_gym


TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
POLICY_DIR = os.path.join(os.getcwd(), f"policy_{TIMESTAMP}")
TENSORBOARD_DIR = os.path.join(POLICY_DIR, "tensorboard")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENT_VISUALIZER = os.path.join(CURRENT_DIR, "visualize_dual_flagella.py")
CURRENT_SWIMMER = os.path.join(CURRENT_DIR, "swimmer.py")
SHARED_POLICY_ID = "shared_policy"


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return SHARED_POLICY_ID


class TrainingMetricsCallback(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        sub_envs = base_env.get_sub_environments()
        if not sub_envs:
            return

        env_ref = sub_envs[env_index]
        episode.custom_metrics["forward_reward"] = float(getattr(env_ref, "last_forward_reward", 0.0))
        episode.custom_metrics["shape_trend_reward"] = float(getattr(env_ref, "last_shape_trend_reward", 0.0))
        episode.custom_metrics["shape_anchor_penalty"] = float(getattr(env_ref, "last_shape_anchor_penalty", 0.0))
        episode.custom_metrics["err_x"] = float(getattr(env_ref, "last_err_x", 0.0))
        episode.custom_metrics["err_y"] = float(getattr(env_ref, "last_err_y", 0.0))
        episode.custom_metrics["shape_error"] = float(getattr(env_ref, "last_shape_error", 0.0))
        episode.custom_metrics["prev_shape_error"] = float(getattr(env_ref, "last_prev_shape_error", 0.0))
        episode.custom_metrics["trend_weight"] = float(getattr(env_ref, "last_trend_weight", 0.0))
        episode.custom_metrics["anchor_weight"] = float(getattr(env_ref, "last_anchor_weight", 0.0))
        episode.custom_metrics["delta_x"] = float(getattr(env_ref, "last_delta_x", 0.0))
        episode.custom_metrics["delta_y"] = float(getattr(env_ref, "last_delta_y", 0.0))
        episode.custom_metrics["robot_1_action_id"] = float(getattr(env_ref, "last_macro_action", (0, 0))[0])
        episode.custom_metrics["robot_2_action_id"] = float(getattr(env_ref, "last_macro_action", (0, 0))[1])
        episode.custom_metrics["episode_steps"] = float(getattr(env_ref, "ep_step", 0))


def maybe_add_scalar(writer, tag, value, step):
    if isinstance(value, bool):
        writer.add_scalar(tag, int(value), step)
        return
    if isinstance(value, (int, float)):
        writer.add_scalar(tag, value, step)
        return
    if hasattr(value, "item"):
        try:
            writer.add_scalar(tag, float(value.item()), step)
        except Exception:
            return


def write_training_scalars(writer, result, iteration):
    maybe_add_scalar(writer, "training/episode_reward_mean", result.get("episode_reward_mean"), iteration)
    maybe_add_scalar(writer, "training/episode_reward_min", result.get("episode_reward_min"), iteration)
    maybe_add_scalar(writer, "training/episode_reward_max", result.get("episode_reward_max"), iteration)
    maybe_add_scalar(writer, "training/episodes_total", result.get("episodes_total"), iteration)
    maybe_add_scalar(writer, "training/num_env_steps_sampled", result.get("num_env_steps_sampled"), iteration)
    maybe_add_scalar(writer, "training/num_env_steps_trained", result.get("num_env_steps_trained"), iteration)
    maybe_add_scalar(writer, "training/num_agent_steps_sampled", result.get("num_agent_steps_sampled"), iteration)
    maybe_add_scalar(writer, "training/num_agent_steps_trained", result.get("num_agent_steps_trained"), iteration)
    maybe_add_scalar(writer, "training/sampler_results/episode_len_mean", result.get("sampler_results", {}).get("episode_len_mean"), iteration)

    learner_info = result.get("info", {}).get("learner", {}).get(SHARED_POLICY_ID, {})
    for key in ("learner_stats", "stats"):
        stats = learner_info.get(key, {})
        if not isinstance(stats, dict):
            continue
        for name, value in stats.items():
            maybe_add_scalar(writer, f"learner/{name}", value, iteration)

    custom_metrics = result.get("custom_metrics", {})
    if isinstance(custom_metrics, dict):
        for name, value in custom_metrics.items():
            maybe_add_scalar(writer, f"custom_metrics/{name}", value, iteration)


def create_summary_writer(log_dir):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorBoard logging requires the 'tensorboard' package. "
            "Install it with 'pip install tensorboard' before training."
        ) from exc
    return SummaryWriter(log_dir=log_dir)


def build_env_config(cli_args, skip_policy_load=False):
    return {
        "forward_ckpt": cli_args.forward_ckpt,
        "cw_ckpt": cli_args.cw_ckpt,
        "ccw_ckpt": cli_args.ccw_ckpt,
        "low_level_hold_steps": swimmer_module.LOW_LEVEL_HOLD_STEPS,
        "macro_horizon": swimmer_module.MACRO_HORIZON,
        "skip_policy_load": skip_policy_load,
    }


def build_ppo_config(cli_args):
    env_stub = swimmer_gym(build_env_config(cli_args, skip_policy_load=True))

    config = ppo.DEFAULT_CONFIG.copy()
    config["env"] = swimmer_gym
    config["num_gpus"] = 0
    config["num_workers"] = 0
    config["num_rollout_workers"] = 0
    config["framework"] = "torch"
    config["env_config"] = build_env_config(cli_args)

    config["gamma"] = 0.995
    config["lr"] = 0.0003
    config["horizon"] = swimmer_module.MACRO_HORIZON
    config["rollout_fragment_length"] = swimmer_module.MACRO_HORIZON
    config["evaluation_duration"] = 10000000

    config["lr_schedule"] = None
    config["use_critic"] = True
    config["use_gae"] = True
    config["lambda_"] = 0.95
    config["kl_coeff"] = 0.2
    config["sgd_minibatch_size"] = 100
    config["train_batch_size"] = 500
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
    config["callbacks"] = TrainingMetricsCallback
    config["disable_env_checking"] = False
    config["multiagent"] = {
        "policies": {
            SHARED_POLICY_ID: (
                None,
                env_stub.observation_space,
                env_stub.action_space,
                {},
            )
        },
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": [SHARED_POLICY_ID],
        "count_steps_by": "env_steps",
    }
    return config


def write_training_run_markdown(run_dir, cli_args, trainer_config, visualizer_snapshot_path, swimmer_snapshot_path):
    env_params = {
        "robot1_init": swimmer_module.ROBOT1_INIT,
        "robot2_init": swimmer_module.ROBOT2_INIT,
        "macro_horizon": swimmer_module.MACRO_HORIZON,
        "low_level_hold_steps": swimmer_module.LOW_LEVEL_HOLD_STEPS,
        "formation_target_dx": swimmer_module.FORMATION_TARGET_DX,
        "formation_target_dy": swimmer_module.FORMATION_TARGET_DY,
        "forward_reward_coef": swimmer_module.FORWARD_REWARD_COEF,
        "shape_error_x_weight": swimmer_module.SHAPE_ERROR_X_WEIGHT,
        "shape_error_y_weight": swimmer_module.SHAPE_ERROR_Y_WEIGHT,
        "shape_trend_reward_coef": swimmer_module.SHAPE_TREND_REWARD_COEF,
        "shape_anchor_penalty_coef": swimmer_module.SHAPE_ANCHOR_PENALTY_COEF,
        "shape_trend_fade_low": swimmer_module.SHAPE_TREND_FADE_LOW,
        "shape_trend_fade_high": swimmer_module.SHAPE_TREND_FADE_HIGH,
        "shape_anchor_near_multiplier": swimmer_module.SHAPE_ANCHOR_NEAR_MULTIPLIER,
        "visualizer_snapshot": visualizer_snapshot_path,
        "swimmer_snapshot": swimmer_snapshot_path,
        "robot_ids": ROBOT_IDS,
        "shared_policy_id": SHARED_POLICY_ID,
        "observation_dim_per_agent": swimmer_module.OBSERVATION_DIM,
        "action_dim_per_agent": len(PRIMITIVE_NAMES),
        "primitive_names": PRIMITIVE_NAMES,
        "reward_mode": "shared team reward copied to both agents",
        "reset_behavior": "hard reset to fixed start poses each episode",
        "historical_plane_mapping": "kept consistent with maintained single-robot branch",
    }

    lines = [
        "# Dual Shared-Policy Training Run Parameters",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Policy directory: `{run_dir}`",
        "",
        "## CLI Parameters",
        "",
        f"- `forward_ckpt`: `{cli_args.forward_ckpt}`",
        f"- `cw_ckpt`: `{cli_args.cw_ckpt}`",
        f"- `ccw_ckpt`: `{cli_args.ccw_ckpt}`",
        f"- `num_cpus`: `{cli_args.num_cpus}`",
        f"- `num_threads`: `{cli_args.num_threads}`",
        "",
        "## PPO / RLlib Parameters",
        "",
        "```python",
        pformat(trainer_config, sort_dicts=True),
        "```",
        "",
        "## Environment Parameters",
        "",
        "```python",
        pformat(env_params, sort_dicts=True),
        "```",
    ]

    with open(os.path.join(run_dir, "TRAINING_PARAMS.md"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def snapshot_current_visualizer(run_dir):
    destination = os.path.join(run_dir, "visualize_dual_flagella.py")
    shutil.copy2(CURRENT_VISUALIZER, destination)
    return destination


def snapshot_current_swimmer(run_dir):
    destination = os.path.join(run_dir, "swimmer.py")
    shutil.copy2(CURRENT_SWIMMER, destination)
    return destination


def main():
    os.makedirs(POLICY_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    visualizer_snapshot_path = snapshot_current_visualizer(POLICY_DIR)
    swimmer_snapshot_path = snapshot_current_swimmer(POLICY_DIR)

    print(f"Policy save dir: {POLICY_DIR}")
    print(f"TensorBoard log dir: {TENSORBOARD_DIR}")
    print(f"Visualizer snapshot: {visualizer_snapshot_path}")
    print(f"Swimmer snapshot: {swimmer_snapshot_path}")
    print(f"Ray CPUs: {args.num_cpus}, PyTorch threads: {args.num_threads}")
    print(f"Forward primitive: {args.forward_ckpt}")
    print(f"CW primitive: {args.cw_ckpt}")
    print(f"CCW primitive: {args.ccw_ckpt}")
    print(os.getcwd())

    ray.init(ignore_reinit_error=True, num_cpus=args.num_cpus)

    config = build_ppo_config(args)
    write_training_run_markdown(POLICY_DIR, args, config, visualizer_snapshot_path, swimmer_snapshot_path)

    trainer = ppo.PPO(config=config, env=swimmer_gym)
    tb_writer = create_summary_writer(TENSORBOARD_DIR)

    now_path = os.getcwd()
    for sub_dir in ("traj", "traj2", "trajp"):
        os.makedirs(os.path.join(now_path, sub_dir), exist_ok=True)

    tb_write_interval = 6

    for i in range(2000):
        print(i)
        result = trainer.train()
        if i % tb_write_interval == 0:
            write_training_scalars(tb_writer, result, i)
            tb_writer.flush()
        print(pretty_print(result))
        if i % 3 == 0:
            ckpt_dir = osp.join(POLICY_DIR, str(i))
            os.makedirs(ckpt_dir, exist_ok=True)
            trainer.save(ckpt_dir)

    tb_writer.close()


if __name__ == "__main__":
    main()
