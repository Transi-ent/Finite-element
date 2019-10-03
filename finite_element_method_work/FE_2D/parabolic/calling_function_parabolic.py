#triangle element

import function_parabolic as f
import numpy as np

def U_analy(x,y,t):
    return np.exp(x+y+t)

def get_error(N1,N2,u,t):
    N=2*N1*N2
    err_lis=[]
    for n in range(N):
        lis=[]
        for i in range(3):   # Nlb=3
            index=f.Tb_function(N1,N2,i,n)
            x,y=f.Pb_function(N1,N2,index-1)
            e=abs(U_analy(x,y,t)-f.wx(N1,N2,n,u))
            lis.append(e)
        err_lis.append(max(lis))
    return max(err_lis)

def getResultList():
    result=[]
    lis=[8,16,32,64]
    for j in range(4):
        i=lis[j]
        A=f.get_stiffness_mat(i,i)
        b=f.get_load_vector(i,i)
        if np.linalg.det(A)!=0:
            inv_A=np.linalg.inv(A)
            u=np.dot(inv_A,b)
            result.append(u)
        else:
            print("Wrong! A is not invertible! when 1/h==",lis[j])

def u_fun(x,y):
    return np.exp(x+y)

t1=1
t0=0
i=4
theta=0.5#TODO: unsure
Nb=(i+1)*(i+1)
X_info=np.zeros((Nb,i+1),float)
delt_t=(t1-t0)/float(i)
for k in range(Nb):
    x,y=f.Pb_function(i,i,k)
    value=u_fun(x,y)
    X_info[k,0]=value
#if bo==0: to obtain A, elif bo==1: to obtain M
A=f.get_stiffness_mat(i,i,0)
M=f.get_stiffness_mat(i,i,1)
for j in range(i):
    b0=f.get_load_vector(i,i,j*delt_t)
    b1=f.get_load_vector(i,i,(j+1)*delt_t)
    print("shape of np.linalg.inv((M/delt_t)+theta*A)",np.linalg.inv((M/delt_t)+theta*A).shape)
    print('shape of b:',b0.shape)
    print('shape of X_info:',X_info.shape)
    X_info[:,j+1]=np.dot(np.linalg.inv((M/delt_t)+theta*A),theta*b1+(1-theta)*b0+(1/delt_t)*np.dot(M,X_info[j])-(1-theta)*np.dot(A,X_info[:,j]))
# if np.linalg.det(A)!=0:
#     inv_A=np.linalg.inv(A)
#     u=np.dot(inv_A,b)
#     err=get_error(i,i,u)
#     print("The solution u is: ",'\n',u)
#     print("The error is: ",'\n',err)
#     # #result.append(u)
# else:
#     print("Wrong! A is not invertible! ")
print('X info matrix:','\n',X_info)
print("Done! ")
