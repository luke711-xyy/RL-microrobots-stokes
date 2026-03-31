import gym, ray
#from gym_particle.envs.swimmer import swimmer_gym
from swimmer_c import swimmer_gym
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.algorithms.ppo import PPO
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.rllib.utils.typing import ModelConfigDict, TensorType
import os
import numpy as np
import math
from os import path

directory_path = os.getcwd()
folder_name = path.basename(directory_path)


# class MyEnv(gym.Env):
#     def __init__(self, env_config):
#         action_space = <gym.Space>
#         observation_space = <gym.Space>
#     def reset(self):
#         return <obs>
#     def step(self, action):
#         return <obs>, <reward: float>, <done: bool>, <info: dict>
cwd = os.path.join(os.getcwd(),"policy")
cwd2 = os.path.join(os.getcwd(),"policy/checkpoint_000000/checkpoint-0")
print(os.getcwd())
ray.init(ignore_reinit_error=True, num_cpus=5)
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
config["framework"]= "torch"
config['gamma']=0.9999
config['lr']=0.0005
config['horizon']=20
config["evaluation_duration"]= 1000000000

config['lr_schedule'] = None
config['use_critic']  = True
config['use_gae']= True
config['lambda_']= 0.95
config['kl_coeff']= 0.2
config['sgd_minibatch_size']= 5
config["train_batch_size"]= 20
config['num_sgd_iter']=5
config['shuffle_sequences']= True
config['vf_loss_coeff']= 1.0
config['entropy_coeff'] = 0.0
config['entropy_coeff_schedule'] = None
config['clip_param']=0.2
config['vf_clip_param']=100000
config['grad_clip']= None
config['kl_target']=0.01






# config['soft_horizon'] = True
# config['no_done_at_end'] =   True

config["evaluation_interval"]=1000000
config["evaluation_duration"]=1

# config["use_lstm"]=True
# config["max_seq_len"]=8
#b=config["model"]
#b["use_lstm"]=True
#b["max_seq_len"]= 5
# config[ "timesteps_per_iteration"]=500

config["min_sample_timesteps_per_iteration"]= 20

directory_path = os.getcwd()
folder_name = path.basename(directory_path)
N=10
wl=1.0
X_ini=0
Y_ini=0
theta_ini=np.random.rand()*2*math.pi
#theta_ini=math.pi/6
state =  np.zeros((N+2),dtype=np.float64)
state[0]=X_ini        
state[1]=Y_ini
state[2]=theta_ini
# state=np.array([-0.38347135 , 1.43589542 , 2.19029877,  1.268934  ,  1.17029603  ,0.85672293\
#   ,1.0121569   ,1.42249195 , 0.55258351],dtype=np.float64)
Xfirst=np.zeros((2),dtype=np.float64)
Xfirst[0]=  X_ini -wl/2*math.cos(state[2])      
Xfirst[1]=  Y_ini-wl/2*math.sin(state[2])
# Xfirst=np.array([0.07317129 ,1.57555878],dtype=np.float64)
XY_positions=np.zeros((N+1,2))
for i in range(N+1):
            
    XY_positions[i,0]=Xfirst[0]+i*wl/N*math.cos(state[2])
    XY_positions[i,1]=Xfirst[1]+i*wl/N*math.sin(state[2])
           
np.savetxt('state.pt',state , delimiter=',')
np.savetxt('Xfirst.pt',Xfirst, delimiter=',')
np.savetxt('XY_positions.pt',XY_positions, delimiter=',')
  
env.__init__(env,{})
#trainer.restore(cwd2)
#trainer = config.build(env=env)

now_path=os.getcwd()
#store_path="/home/users/nus/e0546056/scratch/filament_constriction/bgflow"
#store_folder=os.path.join(store_path,folder_name)
#if os.path.isdir(store_folder)<1:
#    os.mkdir(store_folder)
path1 = os.path.join(now_path,'traj')
path2 = os.path.join(now_path, 'traj2')

if os.path.isdir(path1)<1:
    os.mkdir(path1)
    
if os.path.isdir(path2)<1:
    os.mkdir(path2)
trainer= ppo.PPO(config=config, env=env)
for i in range(500):
    print(i)
#     env.reset(env)
    #print(env.X)
#     trainer.config["critic_lr"]=(1e-3)*(1-i/100)
#     trainer.config["actor_lr"]=(1e-3)*(1-i/100)    
#     trainer.config["tau"]=(1e-3)*(1-i/50)   
#     if i>0 and i%5==0:
#         env.__init__(env,{})
#         trainer=None
#         trainer= ppo.PPO(config=config, env=env)
    try:
        result = trainer.train()
        model = trainer.get_policy().model

# Print model summary
        #print(model)
    except:
        pass
#    path = os.path.join(cwd, str(i))
#    if os.path.isdir(path)<0:
#        os.mkdir(path)
#    checkpoint = trainer.save(path)
#     print(i)
#     trainer.evaluate()
#     print("checkpoint saved at", checkpoint)
#trainer.export_policy_model(cwd)
