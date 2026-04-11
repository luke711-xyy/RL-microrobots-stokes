import argparse
import os
from datetime import datetime
from pprint import pformat


parser = argparse.ArgumentParser(description="Train dual-flagella senior policy")
parser.add_argument("--forward_ckpt", type=str, required=True, help="Checkpoint path for the forward primitive policy")
parser.add_argument("--cw_ckpt", type=str, required=True, help="Checkpoint path for the clockwise turn primitive policy")
parser.add_argument("--ccw_ckpt", type=str, required=True, help="Checkpoint path for the counter-clockwise turn primitive policy")
parser.add_argument("--num_cpus", type=int, default=5, help="Number of CPUs for Ray (default: 5)")
parser.add_argument("--num_threads", type=int, default=5, help="Number of PyTorch threads (default: 5)")
args = parser.parse_args()
os.environ["STOKES_NUM_THREADS"] = str(args.num_threads)

import os.path as osp

import ray
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.logger import pretty_print

import swimmer as swimmer_module
from swimmer import MACRO_ACTION_TABLE, swimmer_gym


TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
POLICY_DIR = os.path.join(os.getcwd(), f"policy_{TIMESTAMP}")
TENSORBOARD_DIR = os.path.join(POLICY_DIR, "tensorboard")


class TrainingMetricsCallback(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        sub_envs = base_env.get_sub_environments()
        if not sub_envs:
            return

        env_ref = sub_envs[env_index]
        episode.custom_metrics["forward_reward"] = float(getattr(env_ref, "last_forward_reward", 0.0))
        episode.custom_metrics["dx_penalty"] = float(getattr(env_ref, "last_dx_penalty", 0.0))
        episode.custom_metrics["dy_penalty"] = float(getattr(env_ref, "last_dy_penalty", 0.0))
        episode.custom_metrics["delta_x"] = float(getattr(env_ref, "last_delta_x", 0.0))
        episode.custom_metrics["delta_y"] = float(getattr(env_ref, "last_delta_y", 0.0))
        episode.custom_metrics["macro_action_id"] = float(getattr(env_ref, "last_macro_action", 0))
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

    learner_info = result.get("info", {}).get("learner", {}).get("default_policy", {})
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
    config = ppo.DEFAULT_CONFIG.copy()
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
    config["callbacks"] = TrainingMetricsCallback
    return config


def write_training_run_markdown(run_dir, cli_args, trainer_config):
    env_params = {
        "robot1_init": swimmer_module.ROBOT1_INIT,
        "robot2_init": swimmer_module.ROBOT2_INIT,
        "macro_horizon": swimmer_module.MACRO_HORIZON,
        "low_level_hold_steps": swimmer_module.LOW_LEVEL_HOLD_STEPS,
        "formation_target_dx": swimmer_module.FORMATION_TARGET_DX,
        "formation_target_dy": swimmer_module.FORMATION_TARGET_DY,
        "forward_reward_coef": swimmer_module.FORWARD_REWARD_COEF,
        "delta_x_penalty_coef": swimmer_module.DELTA_X_PENALTY_COEF,
        "delta_y_penalty_coef": swimmer_module.DELTA_Y_PENALTY_COEF,
        "observation_dim": 14,
        "macro_action_num": len(MACRO_ACTION_TABLE),
        "macro_action_table": MACRO_ACTION_TABLE,
        "low_level_primitives": swimmer_module.PRIMITIVE_NAMES,
        "reset_behavior": "reset-free across episode boundaries",
        "historical_plane_mapping": "kept consistent with maintained single-robot branch",
    }

    lines = [
        "# Dual Senior Training Run Parameters",
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


def main():
    os.makedirs(POLICY_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)

    print(f"Policy save dir: {POLICY_DIR}")
    print(f"TensorBoard log dir: {TENSORBOARD_DIR}")
    print(f"Ray CPUs: {args.num_cpus}, PyTorch threads: {args.num_threads}")
    print(f"Forward primitive: {args.forward_ckpt}")
    print(f"CW primitive: {args.cw_ckpt}")
    print(f"CCW primitive: {args.ccw_ckpt}")
    print(os.getcwd())

    ray.init(ignore_reinit_error=True, num_cpus=args.num_cpus)

    # 训练一开始就把所有高层关键参数写入 policy 根目录，便于后续复现实验。
    config = build_ppo_config(args)
    write_training_run_markdown(POLICY_DIR, args, config)

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
