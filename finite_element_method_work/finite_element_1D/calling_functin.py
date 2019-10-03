import numpy as np
import function as f

list_of_N=[4,8,16,64,128]
left=0
right=1
n_gauss_point=3
for N in list_of_N:
    A=f.get_stiffness_m(N,left,right,n_gauss_point)
    b=f.get_load_vector(N,left,right,n_gauss_point)
    inv_A=np.linalg.inv(A)
    u=np.dot(inv_A,b)
    #print('b=',b)
    print('A=',A)
    #print("when N=",N," the solution is:\n",u,'\n')
    print('—————————————————————————————————————————————————————————————————————————————————————————————————————————————')
