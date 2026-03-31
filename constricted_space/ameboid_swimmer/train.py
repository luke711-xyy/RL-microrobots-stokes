import gym, ray
#from gym_particle.envs.swimmer import swimmer_gym
from swimmer_con import swimmer_gym
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.algorithms.ppo import PPO
import ray.rllib.algorithms.ppo as ppo
from ray.rllib.agents.ppo import PPOTrainer
#from ray.rllib.algorithms.ppo import PPOConfig
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
ray.init(ignore_reinit_error=True, num_cpus=10)
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
# config["actor_hiddens"]=[100, 100]
# config["actor_hidden_activation"]="relu"
# config["critic_hiddens"]=[100, 100]
# config["critic_hidden_activation"]="relu"
# config["use_lstm"]=True
# config["max_seq_len"]= 5
# b=config["model"]
# b["use_lstm"]=True
# b["max_seq_len"]= 5
print(config)
# config[ "timesteps_per_iteration"]=500

config["min_sample_timesteps_per_iteration"]= 20

directory_path = os.getcwd()
folder_name = path.basename(directory_path)
N=20
wl=math.pi
X_ini=-3
Y_ini=1
theta_ini=(0.5+1.0/N)*math.pi
state =  2* np.ones((N+3),dtype=np.float64)*math.pi/N
state[0]=X_ini        
state[1]=Y_ini
state[2]=theta_ini
# state=np.array([-0.55928913 , 0.70217443, -1.96226023 ,-1.20826637, -1.1319446  ,-0.54852674,\
#  -0.62251691 ,-0.66464832 ,-0.99370118 ,-1.59165732, -0.85480718 , 0.15491714,\
#   1.49478047,  1.50688259 , 2.06341986 , 1.99622641 , 2.04925178 , 1.96485323,\
#   2.45242543  ,2.37400877 , 2.89377462,  3.39464889,  4.32092507],dtype=np.float64)
Xfirst=np.zeros((2),dtype=np.float64)
Xfirst[0]=  X_ini +wl/(2*N)/math.sin(math.pi/N)       
Xfirst[1]=  Y_ini
# Xfirst=np.array([-1.04533222 , 1.2861807 ],dtype=np.float64)
XY_positions=np.zeros((N,2))
for i in range(N):
            
    XY_positions[i,0]=X_ini+wl/N/(2)/math.cos(math.pi/N)*math.cos(i*2*math.pi/N)
    XY_positions[i,1]=Y_ini+wl/N/(2)/math.cos(math.pi/N)*math.sin(i*2*math.pi/N)  
# XY_positions=np.array([[-1.0453322235418012, 1.2861806956019362],\
# [-1.1140097814023486, 1.1197974240680733],\
# [-1.0501744221473268, 0.9514969579107204],\
# [-0.9736923706437232, 0.7885537291535851],\
# [-0.8200995132957949, 0.694696208561472],\
# [-0.6738650887822519 ,0.5897414893384046],\
# [-0.5321810147497927 ,0.4787206662214567],\
# [-0.43397448540711675, 0.32787147797549326],\
# [-0.4377291921797234 ,0.14791064285559352],\
# [-0.3195836731765976, 0.012110656333200798],\
# [-0.14173929568774074, 0.039884338040907255],\
# [-0.12806961529062968, 0.21936453148252588],\
# [-0.11657297405214742, 0.39899700972695035],\
# [-0.2017020117902479, 0.5575941311818654],\
# [-0.2759902731325496, 0.7215491683924917],\
# [-0.358863824819547 ,0.8813364470813241],\
# [-0.42797258173666286, 1.0475410791388302],\
# [-0.5668922330935009, 1.1620021225854744],\
# [-0.6964185255017329, 1.2869938806136627],\
# [-0.8709195150069766, 1.3311459435517727],\
# [-1.0451868343220005, 1.2860804199360922]],dtype=np.float64)           
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
trainer= PPOTrainer(config=config, env=env)

policy = trainer.get_policy()
# policy.model.base_model.summary()
#print(policy.model)
#print(policy.model.base_model.summary())
# Print model summary

for i in range(5000):
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

