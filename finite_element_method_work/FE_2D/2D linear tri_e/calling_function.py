import function as f
import numpy as np

def U_analy(x,y):
    return x*y*(1-0.5*x)*(1-y)*np.exp(x+y)
def get_error(N1,N2,u):
    N=2*N1*N2
    err_lis=[]
    for n in range(N):
        lis=[]
        for i in range(3):   # Nlb=3
            index=f.Tb_function(N1,N2,i,n)
            x,y=f.Pb_function(N1,N2,index-1)
            e=abs(U_analy(x,y)-f.wx(N1,N2,n,u))
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


i=8
A=f.get_stiffness_mat(i,i)
b=f.get_load_vector(i,i)
if np.linalg.det(A)!=0:
    inv_A=np.linalg.inv(A)
    u=np.dot(inv_A,b)
    err=get_error(i,i,u)
    print("The solution u is: ",'\n',u)
    print("The error is: ",'\n',err)
    # #result.append(u)
else:
    print("Wrong! A is not invertible! ")

print("Done! ")
