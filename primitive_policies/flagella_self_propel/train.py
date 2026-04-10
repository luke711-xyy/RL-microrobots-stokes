import os
import argparse
from datetime import datetime
from pprint import pformat

# Parse args BEFORE importing swimmer/calculate_v, so env var is set in time
parser = argparse.ArgumentParser(description="Train flagella self-propel swimmer")
parser.add_argument("--num_cpus", type=int, default=5, help="Number of CPUs for Ray (default: 5)")
parser.add_argument("--num_threads", type=int, default=5, help="Number of PyTorch threads (default: 5)")
args = parser.parse_args()
os.environ["STOKES_NUM_THREADS"] = str(args.num_threads)

import gym, ray
#from gym_particle.envs.swimmer import swimmer_gym
import swimmer as swimmer_module
from swimmer import swimmer_gym
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.algorithms.ppo import PPO
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.logger import pretty_print
from ray.rllib.utils.typing import ModelConfigDict, TensorType
import numpy as np
import math
from os import path

class TrainingMetricsCallback(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        sub_envs = base_env.get_sub_environments()
        if not sub_envs:
            return
        env_ref = sub_envs[env_index]
        current_episode = int(getattr(env_ref, "episode_count", 0)) + 1
        if current_episode % 10 != 0:
            return
        episode.custom_metrics["pressure_reward"] = float(getattr(env_ref, "last_pressure_reward", 0.0))
        episode.custom_metrics["direction_penalty_local"] = float(getattr(env_ref, "last_direction_penalty", 0.0))
        episode.custom_metrics["direction_penalty_total"] = float(getattr(env_ref, "last_total_direction_penalty", 0.0))
        episode.custom_metrics["drift_bias_penalty"] = float(getattr(env_ref, "last_drift_bias_penalty", 0.0))
        episode.custom_metrics["recent_displacement"] = float(getattr(env_ref, "last_recent_displacement", 0.0))
        episode.custom_metrics["cumulative_drift_deg"] = float(getattr(env_ref, "last_cumulative_drift_deg", 0.0))
        episode.custom_metrics["local_turn_deg"] = float(getattr(env_ref, "last_signed_turn_deg", 0.0))
        episode.custom_metrics["displacement_scale"] = float(getattr(env_ref, "last_displacement_scale", 0.0))
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


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
cwd = os.path.join(os.getcwd(), f"policy_{timestamp}")
cwd2 = os.path.join(os.getcwd(),"policy/checkpoint_000000/checkpoint-0")
tb_dir = os.path.join(cwd, "tensorboard")
print(f"Policy save dir: {cwd}")
print(f"TensorBoard log dir: {tb_dir}")
print(f"Ray CPUs: {args.num_cpus}, PyTorch threads: {args.num_threads}")
print(os.getcwd())
os.makedirs(cwd, exist_ok=True)
os.makedirs(tb_dir, exist_ok=True)
ray.init(ignore_reinit_error=True, num_cpus=args.num_cpus)
# trainer = ppo.PPOTrainer(env=swimmer_gym, config={
#     "env_config": {},  # config to pass to env class
# })

# while True:
#     print(trainer.train())
#     
env=swimmer_gym
#BaseEnv.to_base_env(env)
# obs, rewards, dones, infos, off_policy_actions = env.poll() 
# 
# print(obs)     


#config = PPOConfig()
config =ppo.DEFAULT_CONFIG.copy()
#config = config.training()
config["num_gpus"] = 0
config["num_workers"] = 0
config["num_rollout_workers"]=0
config["batch_mode"] = "complete_episodes"
config["rollout_fragment_length"] = 3000
config["framework"]= "torch"
config['gamma']=0.9999
config['lr']=0.0003
config['horizon']=3000
config["evaluation_duration"]= 10000000

config['lr_schedule'] = None
config['use_critic']  = True
config['use_gae']= True
config['lambda_']= 0.98
config['kl_coeff']= 0.2
config['sgd_minibatch_size']= 256
config["train_batch_size"]= 6000
config['num_sgd_iter']= 15
config['shuffle_sequences']= True
config['vf_loss_coeff']= 1.0
config['entropy_coeff'] = 0.001
config['entropy_coeff_schedule'] = None
config['clip_param']=0.1
config['vf_clip_param']=100000
config['grad_clip']= None
config['kl_target']=0.01






# config['soft_horizon'] = True
# config['no_done_at_end'] =   True

config["evaluation_interval"]=1000000
config["evaluation_duration"]=1
# config["actor_hiddens"]=[100, 100]
# config["actor_hidden_activation"]="relu"
# config["critic_hiddens"]=[100, 100]
# config["critic_hidden_activation"]="relu"
config["use_lstm"]=True
config["max_seq_len"]= 100

# config[ "timesteps_per_iteration"]=500

config["min_sample_timesteps_per_iteration"]= 6000
config["callbacks"] = TrainingMetricsCallback


def write_training_run_markdown(run_dir, cli_args, trainer_config, env_preview):
    env_params = {
        "N": swimmer_module.N,
        "DT": swimmer_module.DT,
        "ACTION_LOW": swimmer_module.ACTION_LOW,
        "ACTION_HIGH": swimmer_module.ACTION_HIGH,
        "initial_centroid_x": env_preview.X_ini,
        "initial_centroid_y": env_preview.Y_ini,
        "betamax": env_preview.betamax,
        "betamin": env_preview.betamin,
        "centroid_history_window": env_preview.centroid_history.maxlen,
        "displacement_gate_ref": env_preview.displacement_gate_ref,
        "pressure_reward_coef": swimmer_module.PRESSURE_REWARD_COEF,
        "direction_reward_base_coef": swimmer_module.DIRECTION_REWARD_BASE_COEF,
        "direction_window_steps": swimmer_module.DIRECTION_WINDOW_STEPS,
        "drift_bias_reward_coef": swimmer_module.DRIFT_BIAS_REWARD_COEF,
        "drift_bias_segments": swimmer_module.DRIFT_BIAS_SEGMENTS,
        "drift_bias_total_window_steps": swimmer_module.DRIFT_BIAS_TOTAL_WINDOW_STEPS,
        "direction_term_structure": "local_unsigned_penalty + cumulative_signed_drift_penalty",
        "true_centroid_tracking": True,
        "reset_behavior": "reset-free",
    }
    lines = [
        "# Training Run Parameters",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Policy directory: `{run_dir}`",
        "",
        "## CLI Parameters",
        "",
        f"- `num_cpus`: `{cli_args.num_cpus}`",
        f"- `num_threads`: `{cli_args.num_threads}`",
        "",
        "## PPO / RLlib Parameters",
        "",
        "```python",
        pformat(trainer_config, sort_dicts=True),
        "```",
    ]

    lines.extend(
        [
            "",
            "## Environment Parameters",
            "",
            "```python",
            pformat(env_params, sort_dicts=True),
            "```",
        ]
    )

    output_path = os.path.join(run_dir, "TRAINING_PARAMS.md")
    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")

directory_path = os.getcwd()
folder_name = path.basename(directory_path)


# N=int(int(folder_name))
# N=3
# X_ini = 0.0
# Y_ini = 0.3*N
# 
# Xfirst=np.zeros((2),dtype=np.float64)
# Xfirst[1]=  Y_ini      
# state = np.zeros(N+1,dtype=np.float64)
#         #self.state[0]=X_ini         
# state[0]=Y_ini 
# 
# X=0
# Y=0
# m=np.zeros((1,2))
# m[:,0]=X
# m[:,1]=Y       
# mm=np.zeros((1,2))
# mm=Xfirst.copy()
# np.savetxt('state.pt',state , delimiter=',')
# np.savetxt('XY.pt',m, delimiter=',')
# np.savetxt('Xfirst.pt',mm, delimiter=',')
#config["train_batch_size"]=100

#config["num_workers"]= 0
# config["env"]=env
# config["env_config"]={}

#config = config.resources(num_gpus=0)
#config = config.rollouts(num_rollout_workers=1)
#trainer = config.build()
#trainer = PPO()
#print(config.to_dict())   
env_preview = swimmer_gym({})
write_training_run_markdown(cwd, args, config, env_preview)
env.__init__(env,{})
#trainer.restore(cwd2)
#trainer = config.build(env=env)
trainer= ppo.PPO(config=config, env=env)
tb_writer = create_summary_writer(tb_dir)
now_path=os.getcwd()
path1 = os.path.join(now_path,'traj')
path2 = os.path.join(now_path, 'traj2')
pathp = os.path.join(now_path,'trajp')
if os.path.isdir(path1)<1:
    os.mkdir(path1)
    
if os.path.isdir(path2)<1:
    os.mkdir(path2)

if os.path.isdir(pathp)<1:
    os.mkdir(pathp)
#i=199
#print(i)
#cwd_restore = os.path.join(os.getcwd(),"policy2",str(i),"checkpoint_000200")
#trainer.restore(cwd_restore)

for i in range(2000):
    print(i)
#     env.reset(env)
    #print(env.X)
#     trainer.config["critic_lr"]=(1e-3)*(1-i/100)
#     trainer.config["actor_lr"]=(1e-3)*(1-i/100)    
#     trainer.config["tau"]=(1e-3)*(1-i/50)   
#     if i>0 and i%50==0:
#         env.__init__(env,{})
#         trainer=None
#         trainer= ppo.PPO(config=config, env=env)    
    result = trainer.train()
    write_training_scalars(tb_writer, result, i)
    tb_writer.flush()
    print(pretty_print(result))
    if i%10==0:
        path = os.path.join(cwd, str(i))
        os.makedirs(path, exist_ok=True)
        checkpoint = trainer.save(path)
#     print(i)
#     trainer.evaluate()
    #print("checkpoint saved at", checkpoint)
#trainer.export_policy_model(cwd)
tb_writer.close()
