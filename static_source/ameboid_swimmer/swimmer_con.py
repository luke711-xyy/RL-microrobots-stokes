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
from ray.rllib.policy.policy import Policy
#import numba
#from numba import jit, njit
from calculate_v import RK, cal_remaining_w
directory_path = os.getcwd()
folder_name = path.basename(directory_path)
# i=150
# cwd = os.path.join(os.getcwd(),"policy2",str(i),"checkpoint_000151/policies/default_policy")
# my_restored_policy = Policy.from_checkpoint(cwd)

i=810
cwd = os.path.join(os.getcwd(),"policy2",str(i),"checkpoint_000811/policies/default_policy")
my_restored_policy1 = Policy.from_checkpoint(cwd)
i=800
cwd = os.path.join(os.getcwd(),"policy2",str(i),"checkpoint_000801/policies/default_policy")
my_restored_policy2 = Policy.from_checkpoint(cwd)
i=790
cwd = os.path.join(os.getcwd(),"policy2",str(i),"checkpoint_000791/policies/default_policy")
my_restored_policy3 = Policy.from_checkpoint(cwd)
#N=int(int(folder_name))
N=20
wl=1.0
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
traj5=[]
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
        #self.action_space = spaces.Box(low=0, high=1, shape=((6),), dtype=np.float64)
        #self.observation_space = spaces.Box(low=-10000, high=10000,shape=(N,), dtype=np.float64)
        self.observation_space = spaces.Discrete(N)
        #self.observation_space = spaces.MultiDiscrete(np.ones(N)*N)
        self.action_space = spaces.Discrete(3)        
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
        self.X_ini=0
        self.Y_ini=0
        
        
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
        self.Xfirst[0]=  self.X_ini +wl/(2*N)/math.sin(math.pi/N)       
        self.Xfirst[1]=  self.Y_ini 
        
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
        self.itaa=0
        self.reach_targets = np.zeros(self.ntargets, dtype=int)
        self.it = 0
        self.traj=[]
        self.traj2=[]
        self.done=False
        self.pressure_diff=0
        self.XY_positions=np.zeros((N,2))
        for i in range(N):
            
            self.XY_positions[i,0]=self.X_ini+wl/N/(2)/math.cos(math.pi/N)*math.cos(i*2*math.pi/N)
            self.XY_positions[i,1]=self.Y_ini+wl/N/(2)/math.cos(math.pi/N)*math.sin(i*2*math.pi/N)  
            
            
            
        self.C_records=np.zeros((N))
        self.pressure_index=np.ones((N))
    def softmax(self,x):

        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)     
        
        

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
        CcenterX=4
        CcenterY=0
        CcenterX1=0
        CcenterY1=4
        Cf=0.2
        if self.it==0:
            self.state=np.loadtxt('state.pt', delimiter=',')
            self.XY_positions=np.loadtxt('XY_positions.pt', delimiter=',')
            self.Xfirst=np.loadtxt('Xfirst.pt', delimiter=',')
            
            con1=1/np.sqrt((np.squeeze(self.XY_positions[:,0])-CcenterX)**2+(np.squeeze(self.XY_positions[:,1])-CcenterY)**2)
            con2=Cf/np.sqrt((np.squeeze(self.XY_positions[:,0])-CcenterX1)**2+(np.squeeze(self.XY_positions[:,1])-CcenterY1)**2)            
            con=np.squeeze(con1+con2)
            self.Clist=con.copy()
            self.con=np.sum(self.Clist) 
     

               


        global traj
        global traj2
        global traj3
        global traj4
        global traj5                 
        self.reward=0
      
        
        
        
#         probability = self.softmax(action)
#         self.order=int(np.random.choice(numpy.arange(N), p=probability))
        
#         print(probability,self.order,np.argmax(probability))
        self.order+=int(action-1)
        self.order=(self.order+N)%N
        print(self.order,action,int(action-1))
#         w_tmp  = np.clip(actionx, ACTION_LOW, ACTION_HIGH) ## compared clip, tanh is much better for control purpose.
#         w_tmp  =    np.roll(w_tmp, -int(self.order))
 
#         state_predict=self.state.copy()
#         state_predict[3:]+=cal_remaining_ww_tmp*0.2
#         for i in range(N):
#             if abs(state_predict[i+3]-2*math.pi/N)>self.betamax:            
#                  w_tmp=   np.clip(actionx, 0, 0)
#                  break
            
            

    
        

       
    

        
        self.it += 1
        my_restored_policy =None
        my_restored_policy =Policy.from_checkpoint(cwd)
        
        for _ in range(20):
            self.itc+=1
            reward=0
            self.Clist_n=self.Clist.copy()
            self.stateall=self.state[3:].copy()
            self.stateall=np.roll(self.stateall, -int(self.order))             
            #self.Xn=self.X+0.0
            self.state_n=self.state.copy()
        
            Xfirst=self.Xfirst.copy()            
            XY_positions=self.XY_positions.copy()
            Xpositions=np.squeeze(self.XY_positions[:,0].copy())
            Ypositions=np.squeeze(self.XY_positions[:,1].copy())            
            P_list=np.zeros(N)
            action_drive1=my_restored_policy1.compute_single_action(self.stateall)
            action_real1=action_drive1[0]
            action_drive2=my_restored_policy2.compute_single_action(self.stateall)
            action_real2=action_drive2[0]            
            action_drive3=my_restored_policy3.compute_single_action(self.stateall)
            action_real3=action_drive3[0]            
            
            action_real1=np.roll(action_real1, int(self.order))
            action_real2=np.roll(action_real2, int(self.order))            
            action_real3=np.roll(action_real3, int(self.order))            
            v1=cal_remaining_w(self.state_n, action_real1)
            v2=cal_remaining_w(self.state_n, action_real2)            
            v3=cal_remaining_w(self.state_n, action_real3)
            
            state_p1=self.state[3:].copy()+v1*0.2
            state_p2=self.state[3:].copy()+v2*0.2          
            state_p3=self.state[3:].copy()+v3*0.2
            p1_done=False
            p2_done=False
            p3_done=False
            action_real=action_real1.copy()


            m=np.zeros((1,2))
            m[:,0]=self.state[0]
            m[:,1]=self.state[1]
            mm=np.zeros((1,2))
            mm=self.Xfirst.copy()
            self.breakwall=False
            self.breaki=False
            self.breakp=False
            for i in range(N):
                if abs(state_p1[i]-2*math.pi/N)>self.betamax:


                    p1_done=True
                    action_real=action_real2.copy()
                    break            
            
            if p1_done==True:
                for i in range(N):
                    if abs(state_p2[i]-2*math.pi/N)>self.betamax:
                        p2_done=True
                        action_real=action_real3.copy()
                        break            
#             action_drive=my_restored_policy.compute_single_action(self.stateall)
#             action_real=action_drive[0]
#             action_real=np.roll(action_real, -int(self.order))
#             v=cal_remaining_w(self.state_n, action_real)
#             state_p=self.state[3:].copy()+v*0.2
#             p_done=False
            if p2_done==True:
                for i in range(N):
                    if abs(state_p3[i]-2*math.pi/N)>self.betamax:
                        p3_done=True
                        break
            #print(p3_done)
#             if p_done==True:
#                 staten ,Xn ,r,x_first_delta,Xpositions,Ypositions,pressure_all=RK(self.state_n,action_real,self.Xfirst)
#                 print(staten[3:]-state_p)
            
            if p3_done==False:
                staten ,Xn ,r,x_first_delta,Xpositions,Ypositions,pressure_all=RK(self.state_n,action_real,self.Xfirst)

                self.state_n=staten.copy()
                P_list=np.array(pressure_all)
                P_list=np.squeeze(P_list)
    #         P_list=np.array(pressure_all)
                XY_positions=np.concatenate((np.array(Xpositions).reshape(-1,1),np.array(Ypositions).reshape(-1,1)),axis=1)
                con_pre=self.con
                con1=1/np.sqrt((np.squeeze(self.XY_positions[:,0])-CcenterX)**2+(np.squeeze(self.XY_positions[:,1])-CcenterY)**2)
                con2=Cf/np.sqrt((np.squeeze(self.XY_positions[:,0])-CcenterX1)**2+(np.squeeze(self.XY_positions[:,1])-CcenterY1)**2)            
                con=np.squeeze(con2+con1)                
                
               

                self.Clist_n=con.copy()
                #con_now=self.Clist_n[self.order]-np.mean(con)
                con_now=np.sum(con)
                
                reward+=(con_now-con_pre).item()*10000000/N
                #reward+=con_now
                self.con=con_now
                
                #self.Xn+=Xn
                Xfirst+=x_first_delta
            
                      
                loop_sign=False
                for i in range(N-1):
                    for j in range(i+1,N):
                        if j%N!=(i+1)%N and (j+1)%N!=i%N:
                            check1=self.check_overlapping(XY_positions[i%N,0],XY_positions[(i+1)%N,0],XY_positions[j%N,0],XY_positions[(j+1)%N,0])
                            check2=self.check_overlapping(XY_positions[i%N,1],XY_positions[(i+1)%N,1],XY_positions[j%N,1],XY_positions[(j+1)%N,1])
                            
                            if check1 is True and check2 is True:
                                
                                check_p=self.check_product(XY_positions[i%N,0],XY_positions[(i+1)%N,0],XY_positions[j%N,0],XY_positions[(j+1)%N,0],\
                                                           XY_positions[i%N,1],XY_positions[(i+1)%N,1],XY_positions[j%N,1],XY_positions[(j+1)%N,1])
                                if check_p is True:
                                   
                                    self.state_n=self.state.copy()
                                    reward=0
                                    Xfirst=self.Xfirst.copy()
                                    self.con =con_pre
                                    self.Clist_n=self.Clist.copy()
                                    loop_sign=True
                                    #print('collapse')
                                break
                    if loop_sign is True:
                        break
                self.break_points=False
                if loop_sign is False :
                    for i in range(N):
                        for j in range(N):
                            if i!=j and (i+1)%N!=j%N and (j+1)%N!=i%N:
                                if math.sqrt((Xpositions[i]-Xpositions[j])**2+(Ypositions[i]-Ypositions[j])**2)<0.6/N:
                                    #print('touch the constriction')                
                                 
                                    reward=-0
                                    self.con =con_pre
                                    
                                    #self.Xn=self.X
                                    Xfirst=self.Xfirst.copy()             
                                    self.state_n=self.state.copy()
                                    self.Clist_n=self.Clist.copy()
                                    self.break_points=True
                                    break
                        if self.break_points:
                            break
  
#                 self.breakwall=False
#                 for i in range(N):
#                     if Ypositions[i]<0.05 or Ypositions[i]>2.95:
#                         print('touch the wall')
#                         reward=-0               
#                         Xfirst=self.Xfirst.copy()
#                         #self.Xn=self.X                 
#                         self.state_n=self.state.copy()
#                         self.con =con_pre
#                         self.Clist_n=self.Clist.copy()
#                         self.breakwall=True
#                         break            
                
#                 self.breaki=False
#                 for i in range(N):
#                     for j in range(51):
#                         if 1.0*math.cos(math.pi*(Xpositions[i]+j/50*(Xpositions[(i+1)%N]-Xpositions[i])))+(Ypositions[i]+j/50*(Ypositions[(i+1)%N]-Ypositions[i]))-2.0>-0.05\
#                            and abs(Xpositions[i]+j/50*(Xpositions[(i+1)%N]-Xpositions[i]))<1:
#                             print('touch the constriction')                
#                          
#                             reward=-0
#                             self.con =con_pre
#                             
#                             #self.Xn=self.X
#                             Xfirst=self.Xfirst.copy()             
#                             self.state_n=self.state.copy()
#                             self.Clist_n=self.Clist.copy()
#                             self.breaki=True
#                             break
#                     if self.breaki:
#                         break
                        
                        
                        
#                 self.breakp=False 
#                 for i in range(N):  
#                     if abs(P_list[i])>1.0:
#                         print('over')
#                         #reward-=0.1
#                         self.breakp=True
#                         break


                      
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
                
#                 mm=np.zeros((1,2))
#                 mm=self.Xfirst.copy()
                
                if traj==[]:
                    traj=self.state_n
                else:
                    traj=np.concatenate((traj.reshape(-1,N+3),self.state_n.reshape(1,-1)),axis=0)
                    
                if traj2==[]:
                    traj2=m
                else:
                    traj2=np.concatenate((traj2.reshape(-1,5),m.reshape(1,-1)),axis=0)
                    
#                 if traj3==[]:
#                     traj3=mm
#                 else:
#                     traj3=np.concatenate((traj3.reshape(-1,2),mm.reshape(1,-1)),axis=0)
                    
#                 if traj4==[]:
#                     traj4=P_list.copy()
#                 else:
#                     traj4=np.concatenate((traj4.reshape(-1,N),P_list.reshape(1,-1)),axis=0)                
    #             if traj4==[]:
    #                 traj4=action
    #             else:
    #                 traj4=np.concatenate((traj4.reshape(-1,N),action.reshape(1,-1)),axis=0)

# 
#             if self.breakwall==True or  self.breaki==True or self.breakp==True:
#                 
#                 break
                
        if     self.it%20==0 :
            path1=os.path.join(directory_path , 'traj')
            path2=os.path.join(directory_path , 'traj2')
            L1=len(os.listdir(path1))
            L2=len(os.listdir(path2))
            path1 = os.path.join(path1,'traj_'+str(L1)+'.pt')
            path2 = os.path.join(path2,'traj2_'+str(L2)+'.pt')          
            np.savetxt(path1, traj, delimiter=',')        
            np.savetxt(path2, traj2, delimiter=',')            
            #self.istore+=1
            traj=[]
            traj2=[]                
                
#         if     self.it%20==0 :
#             np.savetxt('traj.pt', traj, delimiter=',')        
#             np.savetxt('traj2.pt', traj2, delimiter=',')            
#             np.savetxt('traj3.pt', traj3, delimiter=',') 
#             np.savetxt('traj4.pt', traj4, delimiter=',')
# #             np.savetxt('traj5.pt', traj5, delimiter=',')

        if     self.it%20==0 :
            np.savetxt('state.pt',self.state_n , delimiter=',')
            np.savetxt('XY_positions.pt',self.XY_positions, delimiter=',')
            np.savetxt('Xfirst.pt',mm, delimiter=',')   

        if self.it%20==0 :
            print(self.state[:2],self.it,self.reward,self.order,self.Clist.argsort().argsort())
            print(self.state)
            #print(self.Clist)
        #print(self.reward)

#         if traj5==[]:
#             traj5=P_list
#         else:
#             traj5=np.concatenate((traj5.reshape(-1,N),P_list.reshape(1,-1)),axis=0)        
        
#         for i in range(N):
#             if    abs(P_list[i])>1:
#                 self.pressure_index[i]=0
#                 print(P_list[i])
        
#         P_list=np.roll(P_list,int(self.order))        
#         reward+=-(-2*P_list[0]+P_list[2]+2*P_list[3]+P_list[4]-P_list[1]-P_list[5])*10
        
        

#         con_pre=self.con
#         con=self.get_concentraion(XY_positions)
#         con=np.squeeze(con)
#         con_now=np.sum(con)
#         reward+=(con_now-con_pre)*10000
#        
#         self.con=con_now
        


        
        
        
        
   
        

        
        
        
      





        
#         self.reward+=(add_reward)                  
#         self.state = self.state_n.copy()
#         self.X = self.Xn
#         print(Xpositions,Ypositions)

# 

            

#         con_next=self.get_concentraion(self.XY_positions)
#         con_next=np.squeeze(con_next)
#         self.C_records=self.C_records+con_next            
            

#         if abs(pressure_all[-1])>1:
# #             self.done =True
# #             self.Xn=self.X                 
# #             self.state_n=self.state.copy()            
#             self.reward=-100
#             print(pressure_all[-1])

     
        
        
#         self.Y = self.Yn
        #self.reward+=0.9999**(self.it-1)*reward
        #print(self.state[0],self.state[1],self.state[2],self.state[3] )
#         if self.it%200==0 :
#             print(self.state[:2],self.X ,self.it,self.reward,self.order)
#             print(self.state)
# #         if self.it%100==0 and reward>=0:
# #             reward+=self.X*100#done = True
            #print(self.state,self.X,self.Y)
        #if self.it==1000 :
#         self.stateall=self.state[3:].copy()
#         self.stateall=np.roll(self.stateall, int(self.order))            
        #print(self.it,self.pressure_diff,pressure_end,self.state[:2])
        
        return self.order,self.reward,self.done,{}
#.argsort().argsort()



    def reset(self):
        
        CcenterX=4
        CcenterY=0
        CcenterX1=0
        CcenterY1=4        
        Cf=0.2
        self.itc=0
        self.reward=0
#         if self.XY_positions is None:
#             
#             self.order=0
#             print('None')
#         else:

        self.XY_positions=np.loadtxt('XY_positions.pt', delimiter=',')
        con1=1/np.sqrt((np.squeeze(self.XY_positions[:,0])-CcenterX)**2+(np.squeeze(self.XY_positions[:,1])-CcenterY)**2)
        con2=Cf/np.sqrt((np.squeeze(self.XY_positions[:,0])-CcenterX1)**2+(np.squeeze(self.XY_positions[:,1])-CcenterY1)**2)        
        con=np.squeeze(con1+con2)          

        self.Clist=con.copy()
        self.con=np.sum(self.Clist)
        cmax_index=np.argmax(self.Clist)
        self.order=(cmax_index+(np.random.randint(3)-1)+N)%N
#             self.C_records=self.C_records
#             cmax_index=np.argmax(self.C_records)
#             self.order=-cmax_index
#             print(self.C_records)
#         print(self.order,'order')
#         self.order=0
        self.reward=0        
#         self.stateall=self.state[3:].copy()
#         self.stateall=np.roll(self.stateall, int(self.order))         
#         self.C_records=np.zeros((N))
#         self.pressure_in.argsort().argsort()dex=np.ones((N))

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


