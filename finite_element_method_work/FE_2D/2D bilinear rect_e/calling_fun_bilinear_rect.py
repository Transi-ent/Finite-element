import numpy as np
import function_rect_e as f
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#lis=P_function(3,3,2)
#boundaryedges(3,3,3)
i=2
N1=i
N2=i
A=f.get_stiffness_mat(N1,N2)
print("Det(A)= ",np.linalg.det(A))
#print("A is:",'\n',A)
b=f.get_load_vector(N1,N2)
inv_A=np.linalg.inv(A)
u=np.dot(inv_A,b)
err=f.get_err(N1,N2,u)
print("The err is: ",err)

# TODO: PLOT(没有具体解析式无法绘图，直接索引不行)
# fig = plt.figure()
# ax = Axes3D(fig)
# X = np.linspace(-1, 1, N1+1)
# Y = np.linspace(-1, 1, N2+1)
# X, Y = np.meshgrid(X, Y)
# Z = f.check_result(N1,N2,u,X,Y)
# ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='rainbow')
# plt.show()
