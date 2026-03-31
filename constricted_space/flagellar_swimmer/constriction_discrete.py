import numpy as np
import torch
from scipy.optimize import fsolve
import math
from scipy.interpolate import CubicSpline
l=0
a=0.85
b=0.85
# def equations(vars):
#     
#     x, z = vars
#     eq1 = z-(a+b*math.cos(math.pi*x))
#     eq2 =a*x+b*math.sin(math.pi*x)/math.pi+a-l
#     return [eq1, eq2]






N=12
hN=6
N2=2
hN2=1.0

dense_size=int(N/2)*4
sparse_size=int(N/2)*2
dense_size2=int(N2/2)*4
sparse_size2=2


# Xf=np.zeros((sparse_size+1,sparse_size+1),dtype=np.float64)
# Yf=np.zeros((sparse_size+1,sparse_size+1),dtype=np.float64)
# Zf=np.ones((sparse_size+1,sparse_size+1),dtype=np.float64)*3
# Xq=np.zeros((dense_size+1,dense_size+1),dtype=np.float64)
# Yq=np.zeros((dense_size+1,dense_size+1),dtype=np.float64)
# Zq=np.ones((dense_size+1,dense_size+1),dtype=np.float64)*3








Xf_left,Yf_left = np.meshgrid(np.arange(-hN-1,-1 , N/sparse_size), np.arange(-hN2, hN2+0.1*N2/sparse_size2, N2/sparse_size2))
Xq_left,Yq_left = np.meshgrid(np.arange(-hN-1, -1, N/dense_size), np.arange(-hN2, hN2+0.1*N2/dense_size2, N2/dense_size2))
Zf_left=np.ones_like(Xf_left)*2
Zq_left=np.ones_like(Xq_left)*2

Xf_left=Xf_left.reshape(1,-1)
Yf_left=Yf_left.reshape(1,-1)
Zf_left=Zf_left.reshape(1,-1)
Xq_left=Xq_left.reshape(-1,1)
Yq_left=Yq_left.reshape(-1,1)
Zq_left=Zq_left.reshape(-1,1)



Xf_right,Yf_right = np.meshgrid(np.arange(1,hN+1+0.1*N/sparse_size , N/sparse_size), np.arange(-hN2, hN2+0.1*N2/sparse_size2, N2/sparse_size2))
Xq_right,Yq_right = np.meshgrid(np.arange(1,hN+1+0.1*N/dense_size, N/dense_size), np.arange(-hN2, hN2+0.1*N2/dense_size2, N2/dense_size2))
Zf_right=np.ones_like(Xf_right)*2
Zq_right=np.ones_like(Xq_right)*2
Xf_right=Xf_right.reshape(1,-1)
Yf_right=Yf_right.reshape(1,-1)
Zf_right=Zf_right.reshape(1,-1)
Xq_right=Xq_right.reshape(-1,1)
Yq_right=Yq_right.reshape(-1,1)
Zq_right=Zq_right.reshape(-1,1)

x = np.linspace(-1, 1, 500)
y = a+b*np.cos(math.pi*x)
spline = CubicSpline(x, y)
x_fine = np.linspace(-1, 1, 10000)  # very fine x values
y_fine = spline(x_fine)

arc_length = np.sqrt(np.diff(x_fine)**2 + np.diff(y_fine)**2).sum()
num_points = 15
spacing = arc_length / num_points
distances = np.sqrt(np.diff(x_fine)**2 + np.diff(y_fine)**2).cumsum()

x_points = [x_fine[0]]
y_points = [y_fine[0]]

for i in range(1, len(distances)):
    if distances[i] >= spacing * len(x_points):
        x_points.append(x_fine[i])
        y_points.append(y_fine[i])
        if len(x_points) >= num_points:
            break
print(x_points)
constriction_xf = np.array(x_points)
constriction_zf = 2-np.array(y_points)
print(constriction_zf)
constriction_yf=np.arange(-hN2, hN2+0.1*1, 1)
# dl=0.2
# constriction_yf=np.arange(-hN2, hN2+0.1*1, 1)
# constriction_xf=np.arange(0, 2*a+dl*0.1, dl)
# constriction_zf=np.arange(0, 2*a+dl*0.1, dl)



# length_list=np.arange(0, 2*a+dl*0.1, dl)
# # print(length_list)
# print(length_list)
# for i in range(len(length_list)):
#     l=length_list[i]
#     x, z =  fsolve(equations, (0, 0))    
#     constriction_xf[i]=x
#     constriction_zf[i]=2-z    
# print(2-constriction_zf,constriction_xf)
Xf_constriction= np.repeat((constriction_xf).reshape((1,-1)),len(constriction_yf),axis=0)
Zf_constriction= np.repeat((constriction_zf).reshape((1,-1)),len(constriction_yf),axis=0)
Yf_constriction= np.repeat((constriction_yf).reshape((-1,1)),len(constriction_zf),axis=1)
Xf_constriction=Xf_constriction.reshape(1,-1)
Yf_constriction=Yf_constriction.reshape(1,-1)
Zf_constriction=Zf_constriction.reshape(1,-1)

num_points = 31
spacing = arc_length / num_points


x_points2 = [x_fine[0]]
y_points2 = [y_fine[0]]

for i in range(1, len(distances)):
    if distances[i] >= spacing * len(x_points2):
        x_points2.append(x_fine[i])
        y_points2.append(y_fine[i])
        if len(x_points2) >= num_points:
            break

constriction_xq = np.array(x_points2)
constriction_zq = 2-np.array(y_points2)

constriction_yq=np.arange(-hN2, hN2+0.1*0.5, 0.5)

num_points = 100
spacing = arc_length / num_points


x_points3 = [x_fine[0]]
y_points3 = [y_fine[0]]

for i in range(1, len(distances)):
    if distances[i] >= spacing * len(x_points3):
        x_points3.append(x_fine[i])
        y_points3.append(y_fine[i])
        if len(x_points3) >= num_points:
            break
constriction_xpoints = np.array(x_points3)
constriction_zpoints = 2-np.array(y_points3)

# dl=0.1
# constriction_yq=np.arange(-hN2, hN2+0.1*0.5, 0.5)
# constriction_xq=np.arange(0, 2*a+dl*0.1, dl)
# constriction_zq=np.arange(0, 2*a+dl*0.1, dl)
# length_list=np.arange(0, 2*a+dl*0.1, dl)
# for i in range(len(length_list)):
#     l=length_list[i]
#     x, z =  fsolve(equations, (0, 0))    
#     constriction_xq[i]=x
#     constriction_zq[i]=2-z    
# # print(constriction_zq)
# 
# dl=2*a/100
# constriction_xpoints=np.arange(0, 2*a+dl*0.1, dl)
# constriction_zpoints=np.arange(0, 2*a+dl*0.1, dl)
# length_list=np.arange(0, 2*a+dl*0.1, dl)
# for i in range(len(length_list)):
#     l=length_list[i]
#     x, z =  fsolve(equations, (0, 0))    
#     constriction_xpoints[i]=x
#     constriction_zpoints[i]=2-z 







Xq_constriction= np.repeat((constriction_xq).reshape((1,-1)),len(constriction_yq),axis=0)

Zq_constriction= np.repeat((constriction_zq).reshape((1,-1)),len(constriction_yq),axis=0)
Yq_constriction= np.repeat((constriction_yq).reshape((-1,1)),len(constriction_zq),axis=1)



Xq_constriction=Xq_constriction.reshape(-1,1)
Yq_constriction=Yq_constriction.reshape(-1,1)
Zq_constriction=Zq_constriction.reshape(-1,1)







Xf = np.concatenate((Xf_left,Xf_constriction,Xf_right),axis=1)
Yf = np.concatenate((Yf_left,Yf_constriction,Yf_right),axis=1)
Zf = np.concatenate((Zf_left,Zf_constriction,Zf_right),axis=1)
Xq = np.concatenate((Xq_left,Xq_constriction,Xq_right),axis=0)
Yq = np.concatenate((Yq_left,Yq_constriction,Yq_right),axis=0)
Zq = np.concatenate((Zq_left,Zq_constriction,Zq_right),axis=0)


#print(Xf.shape)
# Xf=Xf.reshape(1,-1)
# Yf=Yf.reshape(1,-1)
# Zf=Zf.reshape(1,-1)
# Xq=Xq.reshape(-1,1)
# Yq=Yq.reshape(-1,1)
# Zq=Zq.reshape(-1,1)

Xf_all=np.squeeze(Xf.reshape(1,-1))
Yf_all=np.squeeze(Yf.reshape(1,-1))
Zf_all=np.squeeze(Zf.reshape(1,-1))
Xq_all=np.squeeze(Xq.reshape(-1,1))
Yq_all=np.squeeze(Yq.reshape(-1,1))
Zq_all=np.squeeze(Zq.reshape(-1,1))
Npointsf=Xf_all.shape[0]
Npointsq=Xq_all.shape[0]


Xf_match_q=np.zeros((Npointsf,6),dtype=np.float64)
Yf_match_q=np.zeros((Npointsf,6),dtype=np.float64)
Zf_match_q=np.zeros((Npointsf,6),dtype=np.float64)


Delta_x=np.zeros((Npointsq,Npointsf),dtype=np.float64)
Delta_y=np.zeros((Npointsq,Npointsf),dtype=np.float64)
Delta_z=np.zeros((Npointsq,Npointsf),dtype=np.float64)
Distance=np.zeros((Npointsq,Npointsf),dtype=np.float64)
Min_Distance_Label=np.zeros((Npointsq,Npointsf),dtype=np.int16)
Min_Distance_num=np.zeros((Npointsf),dtype=np.int16)

        #print(Xf[i,j,:,:].shape)
Delta_x[:,:]=Xf[:,:]-Xq[:,:]     
Delta_y[:,:]=Yf[:,:]-Yq[:,:]
Delta_z[:,:]=Zf[:,:]-Zq[:,:]        
Distance[:,:]=Delta_x[:,:]**2+Delta_y[:,:]**2+Delta_z[:,:]**2
        #print(Distance[i,j,:,:].shape)
for k in range(Distance[:,:].shape[0]):
    Min_Distance_Label[k,np.argmin(Distance[k,:])]=1
for m in range(Distance[:,:].shape[1]):
            #Min_Distance_Label[i,j,k,np.argmin(Distance[i,j,k,:])]=1
    Min_Distance_num[m]=np.sum(Min_Distance_Label[:,m])
    selected_x=Min_Distance_Label[:,m]*Xq_all[:]
    selected_y=Min_Distance_Label[:,m]*Yq_all[:]
    selected_z=Min_Distance_Label[:,m]*Zq_all[:]            
           
    Xf_match_q[m,0:Min_Distance_num[m]]=selected_x[np.nonzero(Min_Distance_Label[:,m])]
    Yf_match_q[m,0:Min_Distance_num[m]]=selected_y[np.nonzero(Min_Distance_Label[:,m])]       
    Zf_match_q[m,0:Min_Distance_num[m]]=selected_z[np.nonzero(Min_Distance_Label[:,m])]
    
    
    
Correponding_label=np.zeros((Xf_match_q.shape[0],np.max(Min_Distance_num)),dtype=np.int16)

for m in range(Distance[:,:].shape[1]):
    selected=Min_Distance_Label[:,m]
    Correponding_label[m,0:Min_Distance_num[m]]=selected[np.nonzero(Min_Distance_Label[:,m])]    
    
    
# print(np.max(Min_Distance_num),np.min(Min_Distance_num))
Xf_match_q=Xf_match_q.reshape(Npointsf,np.max(Min_Distance_num))
Yf_match_q=Yf_match_q.reshape(Npointsf,np.max(Min_Distance_num))
Zf_match_q=Zf_match_q.reshape(Npointsf,np.max(Min_Distance_num))
Min_Distance_num=Min_Distance_num.reshape(Npointsf,1)
Xf_all=Xf_all.reshape(Npointsf,1)
Yf_all=Yf_all.reshape(Npointsf,1)
Zf_all=Zf_all.reshape(Npointsf,1)
torch.save( torch.from_numpy(Xf_match_q), 'Xf_match_q_wall.pt')
torch.save(torch.from_numpy(Yf_match_q), 'Yf_match_q_wall.pt')
torch.save(torch.from_numpy(Zf_match_q), 'Zf_match_q_wall.pt')
torch.save(torch.from_numpy(Xf_all), 'Xf_all_wall.pt')
torch.save(torch.from_numpy(Yf_all), 'Yf_all_wall.pt')
torch.save(torch.from_numpy(Zf_all), 'Zf_all_wall.pt')
torch.save(torch.from_numpy(Min_Distance_num), 'Min_Distance_num_wall.pt')
torch.save(torch.from_numpy(Correponding_label), 'Correponding_label_wall.pt')
torch.save( torch.from_numpy(constriction_xpoints), 'constriction_xpoints.pt')
torch.save( torch.from_numpy(constriction_zpoints), 'constriction_zpoints.pt')
# print(Min_Distance_num)
#print(Zf_all.size)
# print(np.max(Min_Distance_num))
# print(np.min(Min_Distance_num))
#         if i ==0 and j==0:
#             #print(Distance[i,j,:,:])
#             print(ind)
#             print(Min_Distance_Label[i,j,:,:])
# Delta_x=Xf-Xq
# Delta_y=Yf-Yq
# print(Delta_x.shape)



