import gym, ray
#from gym_particle.envs.swimmer import swimmer_gym
from swimmer_con import swimmer_gym
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
# class MyEnv(gym.Env):
#     def __init__(self, env_config):
#         self.action_space = <gym.Space>
#         self.observation_space = <gym.Space>
#     def reset(self):
#         return <obs>
#     def step(self, action):
#         return <obs>, <reward: float>, <done: bool>, <info: dict>
cwd = os.path.join(os.getcwd(),"policy")
cwd2 = os.path.join(os.getcwd(),"policy/checkpoint_000000/checkpoint-0")
print(os.getcwd())
ray.init(ignore_reinit_error=True, num_cpus=4)
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
config['lr']=0.001
config['horizon']=20
config["evaluation_duration"]= 10000

config['lr_schedule'] = None
config['use_critic']  = True
config['use_gae']= True
config['lambda_']= 0.98
config['kl_coeff']= 0.2
config['sgd_minibatch_size']= 5
config["train_batch_size"]= 20
config['num_sgd_iter']= 5
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
config["actor_hiddens"]=[100, 100]
config["actor_hidden_activation"]="relu"
config["critic_hiddens"]=[100, 100]
config["critic_hidden_activation"]="relu"
config["use_lstm"]=True
config["max_seq_len"]= 5

# config[ "timesteps_per_iteration"]=500

config["min_sample_timesteps_per_iteration"]= 20

directory_path = os.getcwd()
folder_name = path.basename(directory_path)
N=20
wl=1
X_ini=0
Y_ini=0
theta_ini=np.random.rand()*2*math.pi
state_group=np.loadtxt('initial_state.pt', delimiter=',')
n=np.random.randint(state_group.shape[0])

state =  np.ones((N+3),dtype=np.float64)
state[0]=X_ini        
state[1]=Y_ini
state[2]=theta_ini
state[3:]=state_group[n,3:]



# Xfirst=np.array([-1.04533222 , 1.2861807 ],dtype=np.float64)
XY_positions=np.zeros((N,2),dtype=np.float64)
state_copy=state.copy()
for i in range(N-1):
    state_copy[i+3]+=state_copy[i+2]

      
for i in range(N-1):
    XY_positions[i+1,0]=XY_positions[i,0]+wl/N*np.cos(state_copy[i+2])
    XY_positions[i+1,1]=XY_positions[i,1]+wl/N*np.sin(state_copy[i+2])        
    
X_dis=np.mean(XY_positions[:,0])-X_ini
Y_dis=np.mean(XY_positions[:,1])-Y_ini




Xfirst=np.zeros((2),dtype=np.float64)
Xfirst[0]=  -X_dis     
Xfirst[1]= -Y_dis 
XY_positions[:,0]-=X_dis
XY_positions[:,1]-=Y_dis
np.savetxt('state.pt',state , delimiter=',')
np.savetxt('Xfirst.pt',Xfirst, delimiter=',')
np.savetxt('XY_positions.pt',XY_positions, delimiter=',')
now_path=os.getcwd()
path1 = os.path.join(now_path,'traj')
path2 = os.path.join(now_path, 'traj2')
if os.path.isdir(path1)<1:
    os.mkdir(path1)
    
if os.path.isdir(path2)<1:
    os.mkdir(path2)
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


#config["train_batch_size"]=100

#config["num_workers"]= 0
# config["env"]=env
# config["env_config"]={}

#config = config.resources(num_gpus=0)
#config = config.rollouts(num_rollout_workers=1)
#trainer = config.build()
#trainer = PPO()
#print(config.to_dict())   
env.__init__(env,{})
#trainer.restore(cwd2)
#trainer = config.build(env=env)
trainer= ppo.PPO(config=config, env=env)
for i in range(200):
    print(i)
#     env.reset(env)
    #print(env.X)
#     trainer.config["critic_lr"]=(1e-3)*(1-i/100)
#     trainer.config["actor_lr"]=(1e-3)*(1-i/100)    
#     trainer.config["tau"]=(1e-3)*(1-i/50)   
#     if i>0 and i%100==0:
#         env.__init__(env,{})
#         trainer=None
#         trainer= ppo.PPO(config=config, env=env)    
    result = trainer.train()
#    path = os.path.join(cwd, str(i))
#    if os.path.isdir(path)<0:
#        os.mkdir(path)
#    checkpoint = trainer.save(path)
#     print(i)
#     trainer.evaluate()
#     print("checkpoint saved at", checkpoint)
#trainer.export_policy_model(cwd)
