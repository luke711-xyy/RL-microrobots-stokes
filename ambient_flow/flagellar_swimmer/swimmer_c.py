import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import os
from os import path
import math
import numpy.matlib
from math import sin
from math import cos
import random
import torch
#import numba
#from numba import jit, njit
from calculate_v import RK
from ray.rllib.policy.policy import Policy

directory_path = os.getcwd()
folder_name = path.basename(directory_path)
#store_path="/home/users/nus/e0546056/scratch/filament_constriction/bgflow"
#store_folder=os.path.join(store_path,folder_name)


i=1000
cwd = os.path.join(os.getcwd(),"policy_translate",str(i),"checkpoint_001001/policies/default_policy")
my_restored_policy1 = Policy.from_checkpoint(cwd)
j=1000
cwd = os.path.join(os.getcwd(),"policy_reorien",str(j),"checkpoint_001001/policies/default_policy")
my_restored_policy2 = Policy.from_checkpoint(cwd)
#N=int(int(folder_name))
N=10
M=1
wl=1.0
K_RWD = 4;
L=1
DIST_EPSI_GOAL = 0.004
COEF_EXCEED=-0.1
DIST_EPSI_INLET=0.05 # need to prevent the particle from too close to the inlets.
MAX_STEP=10000
DT = 0.01

ACTION_LOW = -1
ACTION_HIGH = 1
ACTION_MEAN = (ACTION_LOW + ACTION_HIGH)/2



#theta_ini=M/18.0*math.pi
theta_ini1=0
theta_ini2=0
# beta1_ini=0
# beta2_ini=0
# beta3_ini=0
# 2.6589477  0.6732641  0.67776    0.6898408  0.68705076 0.7361291
#  0.733033   0.6177958  0.6838072
beta1_ini1=0
beta2_ini1=0

beta1_ini2=0
beta2_ini2=0



CcenterX=4
CcenterY=4
Ccenter_array=np.array([CcenterX,CcenterY])


traj=[]
traj2=[]
traj3=[]
traj4=[]

d = torch.device("cpu")
dtype = torch.double
 





class swimmer_gym(gym.Env):
    metadata = {
        'render.modes' : ['human'],
        'video.frames_per_second' : 30
    }
   
    def __init__(self, env_config):

        self.max_rate = 1
#         self.betamax = (1.0*math.pi)/6.0
#         self.betamin = -(1.0*math.pi)/6.0
        self.betamax = (2*math.pi)/(N)
        self.betamin = -self.betamax*0.5
        
        self.dt=DT
        self.action_space = spaces.Discrete(3)
        #self.observation_space = spaces.Box(low=-10000, high=10000,shape=(N+1,), dtype=np.float64)
        #self.observation_space = spaces.MultiDiscrete(np.ones(N+1)*(N+1))
        self.observation_space = spaces.Discrete(6)
        self.viewer = None
        self.X = 0
        self.Y = 0

        self.X_ini= -4    
        self.Y_ini=4
        
        
#         self.state=np.loadtxt('state.pt', delimiter=',')
#         XY=np.loadtxt('XY.pt', delimiter=',')
#         self.X=XY[0]
#         self.Y=XY[1]
#         self.Xfirst=np.loadtxt('Xfirst.pt', delimiter=',')
#         self.Xfirst=np.squeeze(self.Xfirst) 
        #self.it=0
#         self.X=0
        self.reward=0
        self.state = np.zeros((N+2),dtype=np.float64)
        self.state[0]=(self.X_ini)        
        self.state[1]=self.Y_ini
        self.state[2]= 0 
        self.Xfirst=np.zeros((2),dtype=np.float64)
        self.Xfirst[0]=  self.X_ini -wl/2*math.cos(self.state[2])       
        self.Xfirst[1]=  self.Y_ini -wl/2*math.sin(self.state[2])        
        
        #self.state[0]= (X_ini+N/2)
        #self.state[1]= Y_ini        
        
#         self.state[0]=X_ini1         
#         self.state[1]=Y_ini1         
#         self.state[2]=X_ini2
#         self.state[3]=Y_ini2        
        self.dist_epsi_goal = DIST_EPSI_GOAL
        self.ACTION_LOW = ACTION_LOW
        self.ACTION_HIGH = ACTION_HIGH
        self.ACTION_MEAN = (self.ACTION_LOW + self.ACTION_HIGH)/2.0
        #self.seed()
        #self.state = np.array([theta_ini,beta1_ini,beta2_ini,beta3_ini,self.beta4_ini,self.beta5_ini,self.beta6_ini,self.beta7_ini,self.beta8_ini],dtype=np.float64)

#        print('herer')
#        print(self.x_tar)

        self.it = 0
        self.traj=[]
        self.traj2=[]
        self.done=False
        self.pressure_diff=0
        self.istore=0
        self.compare=0
        Xp=np.zeros((N+1),dtype=np.float64)
        Yp=np.zeros((N+1),dtype=np.float64)
        for i in range(N+1):
            Xp[i]=self.Xfirst[0]+i*wl/N*math.cos(self.state[2])
            Yp[i]=self.Xfirst[1]+i*wl/N*math.sin(self.state[2])
            
        self.XY_positions=np.concatenate(((Xp).reshape(-1,1),(Yp).reshape(-1,1)),axis=1)            
        #self.order=np.array([0]).astype(int)
        self.order=0

        

    
    
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
 
    def step(self,action):


        if self.it==0:
            self.state=np.loadtxt('state.pt', delimiter=',')
            self.XY_positions=np.loadtxt('XY_positions.pt', delimiter=',')
            self.Xfirst=np.loadtxt('Xfirst.pt', delimiter=',')
            con=1/np.sqrt((np.squeeze(self.XY_positions[:,0])-CcenterX)**2+(np.squeeze(self.XY_positions[:,1])-CcenterY)**2)
            con=np.squeeze(con)
            self.Clist=con.copy()

        self.it += 1
#         done   = False
        reach  = False
        global traj
        global traj2
        global traj3
        global traj4        
        self.reward=0
#         add_reward=0        
        self.order+=(int(action)-1)
        self.order%=6
        print(self.order)    
        if self.order<4:
            itere=5
        else:
            itere=5
        for _ in range(itere):
                      
            reward=0            
            self.Clist_n=self.Clist.copy()
            self.stateall=self.state[3:].copy()
            if self.order==2:
                self.stateall=-self.stateall
            if self.order==5:
                self.stateall=(self.stateall[::-1].copy())              
            if self.order==3:
                self.stateall=(-self.stateall[::-1].copy())                 
            if self.order==4:
                self.stateall=(self.stateall[::-1].copy())             

            self.state_n=self.state.copy()        
            Xfirst=self.Xfirst.copy()            
            XY_positions=self.XY_positions.copy()
            Xpositions=np.squeeze(self.XY_positions[:,0].copy())
            Ypositions=np.squeeze(self.XY_positions[:,1].copy())              
            P_list=np.zeros(N)
            if self.order==1 or self.order==4:
                action_drive1=my_restored_policy1.compute_single_action(self.stateall)
                action_real1=action_drive1[0]
                action_real1=np.clip(action_real1, -1, 1)
            else:
                action_drive1=my_restored_policy2.compute_single_action(self.stateall)
                action_real1=action_drive1[0]
                action_real1=np.clip(action_real1, -1, 1)            
            
            
            if self.order==2:
                action_real1=-action_real1
            if self.order==5:
                action_real1=(action_real1[::-1].copy())        
            if self.order==3:
                action_real1=-(action_real1[::-1].copy())
            if self.order==4:
                action_real1=(action_real1[::-1].copy())                
            state_p1=self.state[3:].copy()+action_real1*0.2
           
            p1_done=False

            action_real=action_real1.copy()
            
            m=np.zeros((1,5))
            m[:,0]=self.state[0]
            m[:,1]=self.state[1]
            self.pressure_end=0
            m[:,2]=float(self.order)
            m[:,3]=self.Xfirst[0]
            m[:,4]=self.Xfirst[1]            

            self.breakwall=False
            self.breaki=False
            self.breakp=False
            
            
            
#             if self.order==1:
#                 for i in range(N-1):
#                     if (state_p1[i])>self.betamax or (state_p1[i])<self.betamin:
# 
#                         p1_done=True
#                   
#                         break            
#                 
# 
#                     
#             
#             
#             if self.order==-1:
#                 for i in range(N-1):
#                     if (state_p1[i])<-self.betamax or (state_p1[i])>-self.betamin:
# 
#                         p1_done=True
#                       
#                         break            
                
            for i in range(N-1):
                if abs(state_p1[i])>self.betamax:
                    p1_done=True
            
            if p1_done==False:                    

                staten ,Xn ,r,x_first_delta,Xpositions,Ypositions,pressure_diff,self.pressure_end=RK(self.state_n,action_real.astype(np.double),self.Xfirst)
                for i in range(N-1):
                    if abs(staten[i+3])>self.betamax:
                        print('over the limits')

                self.state_n=staten.copy()

                XY_positions=np.concatenate((np.array(Xpositions).reshape(-1,1),np.array(Ypositions).reshape(-1,1)),axis=1)
                con_pre=self.con
#                 CcenterX=9
#                 CcenterY=9
                CcenterXc=0
                CcenterYc=6
                factor_con2=0.0
                con1=1/np.sqrt((np.squeeze(self.XY_positions[:,0])-CcenterX)**2+(np.squeeze(self.XY_positions[:,1])-CcenterY)**2)
                con2=factor_con2/np.sqrt((np.squeeze(self.XY_positions[:,0])-CcenterXc)**2+(np.squeeze(self.XY_positions[:,1])-CcenterYc)**2)
                con=np.squeeze(con1+con2)
                self.Clist_n=con.copy()        
                con_now=np.sum(con)               
                reward+=(con_now-con_pre).item()*10000
                self.con=con_now
                Xfirst+=x_first_delta
                
                
    





                self.state = self.state_n.copy()
                self.Clist=self.Clist_n.copy()            
                #self.X = self.Xn
                self.Xfirst=Xfirst.copy()
                self.XY_positions=XY_positions.copy()            
                self.reward+=(reward)

                m=np.zeros((1,5))
                m[:,0]=self.state[0]
                m[:,1]=self.state[1]
                m[:,2]=float(self.order)     
                m[:,3]=self.Xfirst[0]
                m[:,4]=self.Xfirst[1]

        
        

        
                if traj==[]:
                    traj=self.state_n.copy()                  
                else:
                    traj=np.concatenate((traj.reshape(-1,N+2),self.state_n.reshape(1,-1)),axis=0)
                    
                if traj2==[]:
                    traj2=m.copy()
                else:                    
                    traj2=np.concatenate((traj2.reshape(-1,5),m.reshape(1,-1)),axis=0)
                    

                    

                    
               
                    
        if     self.it%100==0 :
            path1=os.path.join(directory_path , 'traj')
            path2=os.path.join(directory_path , 'traj2')
            L1=len(os.listdir(path1))
            L2=len(os.listdir(path2))
            path1 = os.path.join(path1,'traj_'+str(L1)+'.pt')
            path2 = os.path.join(path2,'traj2_'+str(L2)+'.pt')          
            np.savetxt(path1, traj, delimiter=',')        
            np.savetxt(path2, traj2, delimiter=',') 



            traj=[]
            traj2=[]
        if     self.it%1==0 :
            np.savetxt('state.pt',self.state_n , delimiter=',')
            np.savetxt('XY_positions.pt',self.XY_positions, delimiter=',')
            np.savetxt('Xfirst.pt',self.Xfirst.copy() , delimiter=',')            
            np.savetxt('Ccenter.pt',Ccenter_array.copy() ,delimiter=',')
            
        
        
        
#         self.Y = self.Yn
        #self.reward+=0.9999**(self.it-1)*reward
#         print(self.state[3:] )
        if self.it%10==0 :
            print(self.state[:2],self.it,self.reward,self.order)
            print(self.con)
# #         if self.it%100==0 and reward>=0:
# #             reward+=self.X*100#done = True
            #print(self.state,self.X,self.Y)
        #if self.it==1000 :
        
        #print(self.it,self.pressure_diff,pressure_end,self.state[:2])
        if self.Clist[0]>=self.Clist[-1] and self.reward>=0:
            self.compare=0
        elif self.Clist[0]>=self.Clist[-1] and self.reward<0: 
            self.compare=1
        elif self.Clist[0]<self.Clist[-1] and self.reward>=0:
            self.compare=2
        else:
            self.compare=3
        return self.order,self.reward,self.done,{}
        #self.Clist_n.argsort().argsort()


    def reset(self):
        print(self.it)
        if self.it==0:
            self.state=np.loadtxt('state.pt', delimiter=',')
            self.XY_positions=np.loadtxt('XY_positions.pt', delimiter=',')
            self.Xfirst=np.loadtxt('Xfirst.pt', delimiter=',')

      

        #self.it=0
        
        self.reward=0


        self.reward=0

        self.done=False
#         CcenterX=9+(np.random.rand()-0.5)*3
#         CcenterY=0+(np.random.rand()-0.5)*3
#         CcenterX=6
#         CcenterY=0
#         CcenterXc=0
#         CcenterYc=6

        
        con1=1/np.sqrt((np.squeeze(self.XY_positions[:,0])-CcenterX)**2+(np.squeeze(self.XY_positions[:,1])-CcenterY)**2)
        
        con=np.squeeze(con1)   
        self.con=np.sum(con)
        self.Clist=con.copy()  
        if con[0]<=con[-1]:
            self.order=np.random.randint(3)
        else:
            self.order=np.random.randint(3)+3
              


        
     
        return self.order
        
            

    def _get_obs(self):
        return np.concatenate((self.state, self.reach_targets)) 
    
    def render(self):
        pass
        return 

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None



