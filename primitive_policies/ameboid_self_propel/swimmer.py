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
from calculate_v import RK, cal_remaining_w
directory_path = os.getcwd()
folder_name = path.basename(directory_path)


#N=int(int(folder_name))
N=20
M=1
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

theta_ini=(0.5+1.0/N)*math.pi

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





X_tar = [-0.8]
Y_tar = [0.2]

traj=[]
traj2=[]
traj3=[]
traj4=[]
trajp=[]
d = torch.device("cpu")
dtype = torch.double






class swimmer_gym(gym.Env):
    metadata = {
        'render.modes' : ['human'],
        'video.frames_per_second' : 30
    }
   
    def __init__(self, env_config):
        self.ntargets = len(X_tar)
        self.max_rate = 1
#         self.betamax = (1.0*math.pi)/6.0
#         self.betamin = -(1.0*math.pi)/6.0
        self.betamax = (math.pi)/3
        self.betamin = -(math.pi)/3
        
        self.dt=DT
        self.action_space = spaces.Box(low=-1, high=1, shape=((N),), dtype=np.float64)
        self.observation_space = spaces.Box(low=-10000, high=10000,shape=(N,), dtype=np.float64)
        self.viewer = None
        self.X = 0
        self.Y = 0
        self.X_tar = X_tar
        self.Y_tar = Y_tar
#         self.theta_ini1=theta_ini1
#         self.beta1_ini1=beta1_ini1
#         self.beta2_ini1=beta2_ini1
#         self.theta_ini2=theta_ini2
#         self.beta1_ini2=beta1_ini2
#         self.beta2_ini2=beta2_ini2
        #self.state = np.array([X_ini1,Y_ini1,X_ini2,Y_ini2,theta_ini1,beta1_ini1,beta2_ini1,theta_ini2,beta1_ini2,beta2_ini2],dtype=np.float64)
#         self.state = np.zeros((N+2),dtype=np.float64)
#         self.X_ini=-2
#         self.Y_ini=3.5
        self.X_ini=0.0
        self.Y_ini=0.0   
        
        
#         self.state=np.loadtxt('state.pt', delimiter=',')
#         XY=np.loadtxt('XY.pt', delimiter=',')
#         self.X=XY[0]
#         self.Y=XY[1]
#         self.Xfirst=np.loadtxt('Xfirst.pt', delimiter=',')
#         self.Xfirst=np.squeeze(self.Xfirst) 
        #self.it=0
#         self.X=0
        self.reward=0
        self.state =  2* np.ones((N+3),dtype=np.float64)*math.pi/N
        self.state[0]=(self.X_ini)        
        self.state[1]=self.Y_ini
        self.state[2]=theta_ini
        self.Xfirst=np.zeros((2),dtype=np.float64)
        self.Xfirst[0]=  self.X_ini +1/(2*N)/math.sin(math.pi/N)       
        self.Xfirst[1]=  self.Y_ini
#         self.state =  np.array([-0.21080927, -0.06784672 , 1.46498734 , 0.24406124  ,1.01277679  ,1.31702434,\
#   3.20059798  ,4.58043147 , 3.35720346 , 5.06126583 , 5.19945755 , 4.35347487,\
#   3.98825334 , 4.24393065 , 4.84096878  ,4.77861758 , 6.17843681 , 7.52841732,\
#   7.68416222 , 8.50173488 , 7.81234579  ,8.06598509  ,7.74817265],dtype=np.float64) 
#         self.state =  np.array([1.01865296e+01 , 3.16118783e+00 , 2.04399327e+00 , 7.34766943e-01,\
#          -1.59853304e-01 , 2.02559460e-01, -2.09320789e-01,  2.36367553e-02,\
#           7.14581712e-01,  3.36334262e-01 , 7.51535367e-01 , 2.68580829e-01,\
#           5.46007600e-01 , 8.41923196e-04, -3.77758790e-02 , 5.77102069e-01,\
#           6.02867038e-01 ,-1.46015691e-01 , 1.67691995e-01 , 3.51850222e-01,\
#           3.42911534e-01 , 5.91607255e-01 , 6.23276007e-01],dtype=np.float64)
#         self.Xfirst= np.array([13.94731583 , 3.11044553],dtype=np.float64)     
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
        self.x_tar = np.array([self.X_tar, self.Y_tar], dtype=np.float64) ## 2 rows * ntargets columns
#        print('herer')
#        print(self.x_tar)

        self.reach_targets = np.zeros(self.ntargets, dtype=int)
        self.it = 0
        self.istore=0
        self.traj=[]
        self.traj2=[]
        self.done=False
        self.pressure_diff=0
        self.XY_positions=None
        self.C_records=np.zeros((N))
        self.pressure_index=np.ones((N))
        self.order=0
        
        

    
    def check_overlapping(self,x1,x2,x3,x4):
        check_o=False
        if max(x1,x2) >=min(x3,x4) and min(x1,x2) <=max(x3,x4):
            check_o=True
        return check_o
            
    def check_product(self, x1,x2,x3,x4, y1,y2,y3,y4):            
        check_p=False
        p1=(x1-x3)*(y1-y2)-(y1-y3)*(x1-x2)
        p2=(x1-x4)*(y1-y2)-(y1-y4)*(x1-x2)
        p3=(x3-x1)*(y3-y4)-(y3-y1)*(x3-x4)
        p4=(x3-x2)*(y3-y4)-(y3-y2)*(x3-x4)
        if p1*p2<0.1/(N**4) and p3*p4<0.1/(N**4):
            check_p=True
        return check_p
        
        
            
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
 
    def step(self,action):
#         if self.it==0:
#             self.state=np.loadtxt('state.pt', delimiter=',')
#             XY=np.loadtxt('XY.pt', delimiter=',')
#             self.X=XY[0]
#             self.Y=XY[1]
#             self.Xfirst=np.loadtxt('Xfirst.pt', delimiter=',')
#             self.Xfirst=np.squeeze(self.Xfirst) 
#             #self.it=0
#     #         self.X=0
#             self.reward=0        
        self.it += 1
        
       




#         done   = False
        reach  = False
        global traj
        global traj2
        global traj3
        global traj4
        global trajp                
        self.reward=0
        add_reward=0        
        
        
        
        actionx=action.copy()

        w_tmp  = np.clip(actionx, ACTION_LOW, ACTION_HIGH) ## compared clip, tanh is much better for control purpose.
        w_tmp  =    np.roll(w_tmp, -int(self.order))
#         state_predict=self.state.copy()
#         state_predict[3:]+=cal_remaining_ww_tmp*0.2
#         for i in range(N):
#             if abs(state_predict[i+3]-2*math.pi/N)>self.betamax:            
#                  w_tmp=   np.clip(actionx, 0, 0)
#                  break
            
            

    
        
        self.Xn=self.X+0.0
        reward=0
    
            #print(w_tmpx)
        self.state_n=self.state.copy()
        
        
        
        
#         print(w_tmp)
        staten ,Xn ,r,x_first_delta,Xpositions,Ypositions,pressure_all=RK(self.state_n,w_tmp,self.Xfirst)
#         reward+=r*10
#         self.pressure_diff+=pressure_diff
    #         if Xn>0.05:
    #             print(self.state_n,actionx)
        self.state_n=staten.copy()
        self.Xfirst_n=self.Xfirst.copy()
        P_list=np.array(pressure_all)
        
        
        
        
#         if traj5==[]:
#             traj5=P_list
#         else:
#             traj5=np.concatenate((traj5.reshape(-1,N),P_list.reshape(1,-1)),axis=0)        
        
#         for i in range(N):
#             if    abs(P_list[i])>1:
#                 self.pressure_index[i]=0
        
        
        P_list=np.roll(P_list,int(self.order))        
        
        reward+=-(np.mean(P_list[6:15])-(np.sum(P_list[16:])+np.sum(P_list[:5]))/9)*10 
        #print(reward)
        self.XY_positions=np.concatenate((np.array(Xpositions).reshape(-1,1),np.array(Ypositions).reshape(-1,1)),axis=1)
#         con_next=self.get_concentraion(self.XY_positions)
#         con_next=np.squeeze(con_next)
#         self.C_records=self.C_records+con_next
        
        


        
        
        self.Xn+=Xn
        
        #dis1=math.sqrt((self.state_n[0]-self.X_ini)**2+(self.state_n[1]-self.Y_ini)**2)
        #dis2=math.sqrt((self.state[0]-self.X_ini)**2+(self.state[1]-self.Y_ini)**2)
#         if r==0:
#             reward+=-N
#         else:
#         reward+=(((r)/N)*100)      
#         reward+=  pressure_diff.item()*10      
        
        self.Xfirst_n+=x_first_delta
            
       
        self.reward+=(reward.item())
        
        
        
        for i in range(N):
            if abs(self.state_n[i+3]-2*math.pi/N)>self.betamax:
                self.Xn=self.X
                self.state_n=self.state.copy()
                self.reward=0
                self.Xfirst_n=self.Xfirst.copy()
                P_list=np.zeros(N)
                
                break        
        loop_sign=False
        for i in range(N):
            for j in range(N):
                if j%N!=(i+1)%N and (j+1)%N!=i%N and i%N!=j%N:
                    check1=self.check_overlapping(self.XY_positions[i%N,0],self.XY_positions[(i+1)%N,0],self.XY_positions[j%N,0],self.XY_positions[(j+1)%N,0])
                    check2=self.check_overlapping(self.XY_positions[i%N,1],self.XY_positions[(i+1)%N,1],self.XY_positions[j%N,1],self.XY_positions[(j+1)%N,1])
                    
                    if check1 is True and check2 is True:
                        
                        check_p=self.check_product(self.XY_positions[i%N,0],self.XY_positions[(i+1)%N,0],self.XY_positions[j%N,0],self.XY_positions[(j+1)%N,0],\
                                                   self.XY_positions[i%N,1],self.XY_positions[(i+1)%N,1],self.XY_positions[j%N,1],self.XY_positions[(j+1)%N,1])
                        if check_p is True:
                            self.Xn=self.X
                            self.state_n=self.state.copy()
                            self.reward=0
                            self.Xfirst_n=self.Xfirst.copy()
                            P_list=np.zeros(N)
                            loop_sign=True
                        
                        break
            if loop_sign is True:
                break
        
        self.break_points=False
        for i in range(N):
            for j in range(N):
                if i!=j:
                    if math.sqrt((Xpositions[i]-Xpositions[j])**2+(Ypositions[i]-Ypositions[j])**2)<0.6/N:
                                        
                             
                                                                            
                        self.Xn=self.X
                        self.state_n=self.state.copy()
                        self.reward=0
                        self.Xfirst_n=self.Xfirst.copy()
                        P_list=np.zeros(N)
                        self.break_points=True

                        break
                        
            if self.break_points==True:
                break
        self.state = self.state_n.copy()
        self.X = self.Xn
        self.Xfirst=self.Xfirst_n.copy()
#         print(Xpositions,Ypositions)
#         for i in range(N+1):
#             if Ypositions[i]<0.05 or Ypositions[i]>4.95:
#                 
#                 self.done =True
#                 self.reward=-100
#                 print(self.done)
#                 self.Xn=self.X                 
#                 self.state_n=self.state.copy()                
#                 break
# 
# 
# #         for i in range(N+1):
# #             if 1-math.cos(math.pi*Xpositions[i]+math.pi)+Ypositions[i]-5>0.1 and abs(Xpositions[i])<1:
# #                 self.reward-=1
# 
#         for i in range(N+1):
#             if 1-math.cos(math.pi*Xpositions[i]+math.pi)+Ypositions[i]-5>0 and abs(Xpositions[i])<1:
#                 
#                 self.done =True
#                 self.reward=-100
#                 print(self.done)
#                 self.Xn=self.X                 
#                 self.state_n=self.state.copy()                
#                 break         

#         if abs(pressure_all[-1])>1:
# #             self.done =True
# #             self.Xn=self.X                 
# #             self.state_n=self.state.copy()            
#             self.reward=-100
#             print(pressure_all[-1])

        m=np.zeros((1,4))
        m[:,0]=self.state[0]
        m[:,1]=self.state[1]
        m[:,2]=self.Xfirst[0]
        m[:,3]=self.Xfirst[1]        
        

        
        if traj==[]:
            traj=self.state_n
        else:
            traj=np.concatenate((traj.reshape(-1,N+3),self.state_n.reshape(1,-1)),axis=0)
            
        if traj2==[]:
            traj2=m
        else:
            traj2=np.concatenate((traj2.reshape(-1,4),m.reshape(1,-1)),axis=0)
            

        if trajp==[]:
            trajp=P_list
        else:
            trajp=np.concatenate((trajp.reshape(-1,N),P_list.reshape(1,-1)),axis=0)            


        if     self.it%4000==0 :
            path1 = os.path.join(directory_path , 'traj','traj_'+str(self.istore)+'.pt')
            path2 = os.path.join(directory_path , 'traj2','traj2_'+str(self.istore)+'.pt')
            pathp = os.path.join(directory_path , 'trajp','trajp_'+str(self.istore)+'.pt')            
            np.savetxt(path1, traj, delimiter=',')        
            np.savetxt(path2, traj2, delimiter=',')
            np.savetxt(pathp, trajp, delimiter=',')            
            self.istore+=1
            traj=[]
            traj2=[]
            trajp=[]            
            
#         if     self.it%200==0 :
#             np.savetxt('traj.pt', traj, delimiter=',')        
#             np.savetxt('traj2.pt', traj2, delimiter=',')            
#             np.savetxt('traj3.pt', traj3, delimiter=',') 
#             np.savetxt('traj4.pt', traj4, delimiter=',')
#             np.savetxt('trajp.pt', trajp, delimiter=',')
#             np.savetxt('state.pt',self.state_n , delimiter=',')
#             np.savetxt('XY.pt',m, delimiter=',')
#             np.savetxt('Xfirst.pt',mm, delimiter=',')        
        
        
#         self.Y = self.Yn
        #self.reward+=0.9999**(self.it-1)*reward
        #print(self.state[0],self.state[1],self.state[2],self.state[3] )
        if self.it%200==0 :
            print(self.state[:2],self.X ,self.it,self.reward)
            print(self.state)
# #         if self.it%100==0 and reward>=0:
# #             reward+=self.X*100#done = True
            #print(self.state,self.X,self.Y)
        #if self.it==1000 :
        self.stateall=self.state[3:].copy()
        self.stateall=np.roll(self.stateall, int(self.order))            
        #print(self.it,self.pressure_diff,pressure_end,self.state[:2])
        return self.stateall,self.reward,self.done,{}




    def reset(self):
        

        

        self.reward=0
#         self.order=0
#         CcenterX=20/6*N
#         CcenterY=10/6*N
#         if self.XY_positions is None:
#             self.order=0
#             print('None')
#         else:
#             Clist=1/np.sqrt((np.squeeze(self.XY_positions[:,0])-CcenterX)**2+(np.squeeze(self.XY_positions[:,1])-CcenterY)**2)
#             cmax_index=np.argmax(Clist)
#             self.order=-cmax_index            
#             con=self.get_concentraion(self.XY_positions)
#             con=np.squeeze(con)
            #self.C_records=self.C_records
            #cmax_index=np.argmax(self.C_records)
            #self.order=-cmax_index
#             print(self.C_records)
#         print(self.order,'order')

        self.reward=0        
        self.stateall=self.state[3:].copy()
        self.stateall=np.roll(self.stateall, int(self.order))         
        self.C_records=np.zeros((N))
        self.pressure_index=np.ones((N))

        return self.stateall

    def _get_obs(self):
        return np.concatenate((self.state, self.reach_targets)) 
    
    def render(self):
        pass
        return 

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


