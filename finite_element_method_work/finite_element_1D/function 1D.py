import numpy as np
from scipy import sparse

def get_stiffness_m(N,left,right,n_gauss_point):
    """get the stiffness matrix of 1 D finite element"""
    #A = np.random.random(size=(N+1,N+1))
    #for i in range(N+1):
    #    for j in range(N+1):
    #        A[i,j]=0
    A=np.zeros((N+1,N+1),dtype=float)
    X=list(range(N+1))
    h=(right-left)/N
    for i in range(N+1):
        X[i]=i*h

    for n in range(N):
        for alpha in range(2):
            for beta in range(2):
                r=A_gauss_quadrature_1D(alpha,beta,X[n],X[n+1],n_gauss_point)
                A[Tb_function(beta,n,N),Tb_function(alpha,n,N)]+=r

    for j in range(N+1):
        A[0,j]=0
    A[0,0]=1
    for j in range(N+1):
        A[N,j]=0
    A[N,N]=1
    return A

def A_gauss_quadrature_1D(alpha,beta,a,b,n):
    h=b-a
    if alpha==1:
        p_1_a=-1/h
    else:
        p_1_a=1/h

    if beta==1:
        p_1_b=-1/h
    else:
        p_1_b=1/h

    coeff=p_1_a*p_1_b*h/2
    if n==2:
        result=c_f_x(-0.5773503,a,b)*coeff+c_f_x(0.5773503,a,b)*coeff
    elif n==3:
        result=(5/9)*c_f_x(-0.7745967,a,b)*coeff+(5/9)*c_f_x(0.7745967,a,b)*coeff+(8/9)*c_f_x(0,a,b)*coeff

    elif n==4:
        result=0.3478548*c_f_x(-0.8611363,a,b)*coeff+0.3478548*c_f_x(0.8611363,a,b)*coeff+0.6521452*c_f_x(-0.3399810,a,b)*\
               coeff+0.6521452*c_f_x(0.3399810,a,b)*coeff
    elif n==5:
        result=0.2369269*c_f_x(-0.9061798,a,b)*coeff+0.2369269*c_f_x(0.9061798,a,b)*coeff+0.4786287*c_f_x(-0.5384693,a,b)*\
               coeff+0.4786287*c_f_x(0.5384693,a,b)*coeff+0.5688889*c_f_x(0,a,b)*coeff
    else:
        print('number of gauss point needed is too many!!!')

    return result

def Tb_function(t,n,N):                                       # it's ludicrous,ridiculous, it's better to use numpy
    Tb=np.zeros((2,N),dtype=int)
    for i in range(2):
        for j in range(N):
            if i==0:
                Tb[i,j]=j
            elif i==1:
                Tb[i,j]=j+1
    #Tb=[[0],[1]]
    #for i in range(2):
    #    for j in range(1,N):
    #        if i==0:
    #            Tb[0].append(j)
    #        else:
    #            Tb[1].append(j+1)

    result=Tb[t,n]
    return result
def Pb_function(alpha,n_g,N):
    left=0
    right=1
    X=np.linspace(left,right,N+1,dtype=float)
    P=np.zeros((2,N+1))
    P[0,:]=X
    P[1,:]=X+(right-left)/N
    return P[alpha,n_g]

def c_f_x(t,a,b):
    result=np.exp(x_to_t(t,a,b))
    return result

def x_to_t(t,a,b):
    result=(b-a)/2*t+(b+a)/2
    return result

def get_load_vector(N,left,right,n_gp):
    b=np.zeros((N+1,1))
    X=list(range(N+1))
    h=(right-left)/N
    for i in range(N+1):
        X[i]=i*h

    for n in range(N):
        for beta in range(2):
            r=b_gauss_quadrature_1D(beta,X[n],X[n+1],n_gp)
            b[Tb_function(beta,n,N)]+=r
    b[0]=0
    b[N]=np.cos(1)
    return b

def b_gauss_quadrature_1D(beta,a,b,n):
    a1=a
    b1=b
    if beta==0:
        if n==2:
            result=f_f_x(-0.5773503,a1,b1)*posi1(-0.5773503,a1,b1)+f_f_x(0.5773503,a1,b1)*posi1(0.5773503,a1,b1)
        elif n==3:
            result=(5/9)*f_f_x(-0.7745967,a1,b1)*posi1(-0.7745967,a1,b1)+(5/9)*f_f_x(0.7745967,a1,b1)*\
                   posi1(0.7745967,a1,b1)+(8/9)*f_f_x(0,a1,b1)*posi1(0,a1,b1)
        elif n==4:
            result=0.3478548*f_f_x(-0.8611363,a1,b1)*posi1(-0.8611363,a1,b1)+0.3478548*f_f_x(0.8611363,a1,b1)*posi1(0.8611363,a1,b1)+\
                   0.6521452*f_f_x(-0.3399810,a1,b1)*posi1(-0.3399810,a1,b1)+0.6521452*f_f_x(0.3399810,a1,b1)*posi1(0.3399810,a1,b1)
        elif n==5:
            result=0.2369269*f_f_x(-0.9061798,a1,b1)*posi1(-0.9061798,a1,b1)+0.2369269*f_f_x(0.9061798,a1,b1)*posi1(0.9061798,a1,b1)+\
                   0.4786287*f_f_x(-0.5384693,a1,b1)*posi1(-0.5384693,a1,b1)+0.4786287*f_f_x(0.5384693,a1,b1)*posi1(0.5384693,a1,b1)+\
                   0.5688889*f_f_x(0,a1,b1)*posi1(0,a1,b1)
        else:
            print('Number of gauss point needed is too many')
    elif beta==1:
        if n==2:
            result=f_f_x(-0.5773503,a1,b1)*posi2(-0.5773503,a1,b1)+f_f_x(0.5773503,a1,b1)*posi2(0.5773503,a1,b1)
        elif n==3:
            result=(5/9)*f_f_x(-0.7745967,a1,b1)*posi2(-0.7745967,a1,b1)+(5/9)*f_f_x(0.7745967,a1,b1)*posi2(0.7745967,a1,b1)+\
                   (8/9)*f_f_x(0,a1,b1)*posi2(0,a1,b1)
        elif n==4:
            result=0.3478548*f_f_x(-0.8611363,a1,b1)*posi2(-0.8611363,a1,b1)+0.3478548*f_f_x(0.8611363,a1,b1)*posi2(0.8611363,a1,b1)+\
                   0.6521452*f_f_x(-0.3399810,a1,b1)*posi2(-0.3399810,a1,b1)+0.6521452*f_f_x(0.3399810,a1,b1)*posi2(0.3399810,a1,b1)
        elif n==5:
            result=0.2369269*f_f_x(-0.9061798,a1,b1)*posi2(-0.9061798,a1,b1)+0.2369269*f_f_x(0.9061798,a1,b1)*posi2(0.9061798,a1,b1)+\
                   0.4786287*f_f_x(-0.5384693,a1,b1)*posi2(-0.5384693,a1,b1)+0.4786287*f_f_x(0.5384693,a1,b1)*posi2(0.5384693,a1,b1)+\
                   0.5688889*f_f_x(0,a1,b1)*posi2(0,a1,b1)
        else:
            print('Number of gauss point needed is too many')
    else:
        print('wrong beta input')

    return result

def f_f_x(t,a,b):
    x0=x_to_t(t,a,b)
    result=-np.exp(x0)*(np.cos(x0)-2*np.sin(x0)-x0*np.cos(x0)-x0*np.sin(x0))
    return result

def posi1(t,a,b):
    result=(b-x_to_t(t,a,b))/2
    return result
def posi2(t,a,b):
    result=(x_to_t(t,a,b)-a)/2
    return result

def Analytic_s(n,N):
    X=np.array([0]*2,dtype=float)
    for i in range(2):
        index=Tb_function(i,n,N)
        X[i]=Pb_function(i,index,N)
    x0=0.5*(X[0]+X[1])
    return -np.exp(x0)*(np.cos(x0)-2*np.sin(x0)-x0*np.cos(x0)-x0*np.sin(x0))
def Wx(N,u,n_g):
    Nlb=2
    ub=np.array([0]*Nlb,dtype=float)
    X=np.array([0]*2,dtype=float)
    for i in range(2):
        index=Tb_function(i,n_g,N)
        ub[i]=u[index]
        X[i]=Pb_function(i,index,N)
    x0=0.5*(X[0]+X[1])
    return ub[0]*pusi(x0,X,0)+ub[1]*pusi(x0,X,1)
def pusi(x,X,beta):
    if beta==0:
        return (X[1]-x)/(X[1]-X[0])
    elif beta==1:
        return (x-X[0])/(X[1]-X[0])
def get_err(N,u):
    """ Computing the error of middle point of each element """
    err_lis=[]
    for i in range(N):
        Wn=Wx(N,u,i)
        Ux=Analytic_s(i,N)
        err_lis.append(abs(Wn-Ux))
    return max(err_lis)

N=16
left=0
right=1
n_gp=4
A=get_stiffness_m(N,left,right,n_gp)
b=get_load_vector(N,left,right,n_gp)
inv_A=np.linalg.inv(A)
u=np.dot(inv_A,b)
print("The u is: ",'\n',u)
err=get_err(N,u)
print("The error is: ",err)
