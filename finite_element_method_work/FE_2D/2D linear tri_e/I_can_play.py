from scipy import integrate as inte
import numpy as np
import matplotlib.pyplot as plt
import function as f
from mpl_toolkits.mplot3d import Axes3D

def get_position(N1,N2):
    left=-1
    right=1
    bottom=-1
    top=1
    x_array=np.linspace(left,right,N1+1)
    y_array=np.linspace(bottom,top,N2+1)
    return x_array,y_array

def analytic_s(x,y):
    return x*y*(1-0.5*x)*(1-y)*np.exp(x+y)
# x,y=np.mgrid[-1:1:20j, -1:1:20j]
# z=x*y*(1-0.5*x)*(1-y)*np.exp(x+y)
# fig=plt.figure(figsize=(8,6))
# ax=plt.subplot(111,projection='3d')
# ax.plot_surface(x,y,z,rstride=1,cstride=1,cmap='rainbow')
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# plt.show()
i=8
A=f.get_stiffness_mat(i,i)
b=f.get_load_vector(i,i)
if np.linalg.det(A)!=0:
    inv_A=np.linalg.inv(A)
    u=np.dot(inv_A,b)
#     print("The length of u is:",len(u))
#     # err=get_error(i,i,u)
#     # print("The solution u is: ",'\n',u)
#     # print("The error is: ",'\n',err)
#     # #result.append(u)
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     X = np.arange(-1, 1, 0.125)
#     Y = np.arange(-1, 1, 0.125)
#     X, Y = np.meshgrid(X, Y)
#     Z = wx_rectified(i,i,u,X,Y)
#     ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='rainbow')
#     plt.show()
#
# else:
#     print("Wrong! A is not invertible! ")
# #wx_rectified(16,16,u,0,0)
X=np.linspace(-1,1,9)
Y=np.linspace(-1,1,9)
U_analytic=[]
for x in X:
    for y in Y:
        s=analytic_s(x,y)
        U_analytic.append(s)

U_analytic_mat=np.mat(U_analytic)
print("U_analytic_mat is:",'\n',U_analytic_mat,'\n',"The length: ",len(U_analytic_mat[0]))
print("Numerical solution is:",'\n',np.transpose(u),'\n',"the length: ",len(u))
subtraction=U_analytic_mat-np.transpose(u)
error=map(abs,subtraction)
err=list(error)
max_err=max(err)
print("The max error is:",'\n',max_err)
