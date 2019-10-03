import numpy as np

def get_stiffness_mat(N1,N2):
    """
    :param Nlb: You can set the value of Nlb(number of local basis function), Nlb=9 here
    :param N: number of unknowns, or of finite element nodes
    :param N: number of finite elements
    :return: stiffness matrix
    """
    Nlb=9
    Nb=(2*N1+1)*(2*N2+1)
    nbn=4*(N1+N2)
    N=N1*N2
    A=np.zeros((Nb,Nb),dtype=float)           #TODO: taking advantage of sparse matrix may accelerate computing
    for n in range(N):
        for alpha in range(Nlb):
            for beta in range(Nlb):
                r=gauss_quadrature_stiff(N1,N2,n,alpha,beta)
                lis=Tb_function(N1,N2,n)
                A[lis[beta],lis[alpha]]+=r
    # Deal with the boundary nodes
    for k in range(nbn):
        i=boundaryedges(N1,N2,k)
        A[i,:]=0
        A[i,i]=1
    return A
def gauss_quadrature_stiff(N1,N2,n,alpha,beta):
    """compute the integration, phix, phiy is the derivative of pusi(x,y)"""
    Nlb=9
    X=np.array([0]*Nlb,dtype=float)
    Y=np.array([0]*Nlb,dtype=float)
    index_lis=Tb_function(N1,N2,n)
    for i in range(Nlb):
        X[i],Y[i]=Pb_function(N1,N2,index_lis[i])
    h1=X[1]-X[0]
    h2=Y[3]-Y[0]
    r=4/(h1**2)*((25/81)*phix(-0.7745967,-0.7745967,alpha)*phix(-0.7745967,-0.7745967,beta)+
                 (40/81)*phix(-0.7745967,0,alpha)*phix(-0.7745967,0,beta)+
                 (25/81)*phix(-0.7745967,0.7745967,alpha)*phix(-0.7745967,0.7745967,beta)+
                 (40/81)*phix(0,-0.7745967,alpha)*phix(0,-0.7745967,beta)+
                 (64/81)*phix(0,0,alpha)*phix(0,0,beta)+
                 (40/81)*phix(0,0.7745967,alpha)*phix(0,0.7745967,beta)+
                 (25/81)*phix(0.7745967,-0.7745967,alpha)*phix(0.7745967,-0.7745967,beta)+
                 (40/81)*phix(0.7745967,0,alpha)*phix(0.7745967,0,beta)+
                 (25/81)*phix(0.7745967,0.7745967,alpha)*phix(0.7745967,0.7745967,beta)   )+\
      4/(h2**2)*((25/81)*phiy(-0.7745967,-0.7745967,alpha)*phiy(-0.7745967,-0.7745967,beta)+
                 (40/81)*phiy(-0.7745967,0,alpha)*phiy(-0.7745967,0,beta)+
                 (25/81)*phiy(-0.7745967,0.7745967,alpha)*phiy(-0.7745967,0.7745967,beta)+
                 (40/81)*phiy(0,-0.7745967,alpha)*phiy(0,-0.7745967,beta)+
                 (64/81)*phiy(0,0,alpha)*phiy(0,0,beta)+
                 (40/81)*phiy(0,0.7745967,alpha)*phiy(0,0.7745967,beta)+
                 (25/81)*phiy(0.7745967,-0.7745967,alpha)*phiy(0.7745967,-0.7745967,beta)+
                 (40/81)*phiy(0.7745967,0,alpha)*phiy(0.7745967,0,beta)+
                 (25/81)*phiy(0.7745967,0.7745967,alpha)*phiy(0.7745967,0.7745967,beta)   )
    return r


def phix(x,y,beta):
    if beta==0:
        return 0.25*y-0.5*x*y-0.25*y**2+0.5*x*y**2
    elif beta==1:
        return -0.25*y-0.5*x*y+0.25*y**2+0.5*x*y**2
    elif beta==2:
        return 0.25*y+0.5*x*y+0.25*y**2+0.5*x*y**2
    elif beta==3:
        return -0.25*y+0.5*x*y-0.25*y**2+0.5*x*y**2
    elif beta==4:
        return x*y-x*y**2
    elif beta==5:
        return 0.5+x-0.5*y**2-x*y**2
    elif beta==6:
        return -x*y-x*y**2
    elif beta==7:
        return -0.5+x+0.5*y**2-x*y**2
    elif beta==8:
        return -2*x+2*x*y**2
def phiy(x,y,beta):
    if beta==0:
        return 0.25*x-0.5*x*y-0.25*x**2+0.5*y*x**2
    elif beta==1:
        return -0.25*x+0.5*x*y-0.25*x**2+0.5*y*x**2
    elif beta==2:
        return 0.25*x+0.5*x*y+0.25*x**2+0.5*y*x**2
    elif beta==3:
        return -0.25*x-0.5*x*y+0.25*x**2+0.5*y*x**2
    elif beta==4:
        return -0.5+y+0.5*x**2-x**2*y
    elif beta==5:
        return -x*y-y*x**2
    elif beta==6:
        return 0.5+y-0.5*x**2-x**2*y
    elif beta==7:
        return x*y-y*x**2
    elif beta==8:
        return -2*y+2*x**2*y

def Tb_function(N1,N2,n_e):
    Nlb=9
    N=N1*N2
    Tb=np.zeros((Nlb,N),dtype=int)
    Tb[:,0]=[0,2*(2*N2+1),2*(2*N2+1)+2,2,   2*N2+1,2*(2*N2+1)+1,2*N2+3,1,   2*N2+2]
    for i in range(1,N2):
        Tb[:,i]=Tb[:,i-1]+np.array([2]*Nlb)
    for j in range(1,N1):
        Tb[:,j*N2]=Tb[:,j*N2-1]+np.array([2*N2+4]*Nlb)
        for k in range(1,N2):
            Tb[:,j*N2+k]=Tb[:,j*N2+k-1]+np.array([2]*Nlb)

    #print("The Tb is: ",'\n',Tb)
    return Tb[:,n_e]
def Pb_function(N1,N2,n_g):
    Nb=(2*N1+1)*(2*N2+1)
    left=-1
    right=1
    bottom=-1
    top=1
    lis=[]
    P=np.zeros((2,Nb),dtype=float)
    x_array=np.linspace(left,right,2*N1+1,dtype=float)
    y_array=np.linspace(bottom,top,2*N2+1,dtype=float)
    x_lis=list(x_array)
    for i in x_lis:
        lis+=[i]*(2*N2+1)
    P[0,:]=lis
    P[1,:]=list(y_array)*(2*N1+1)
    #print("P is: ",'\n',P)
    x,y=P[:,n_g]
    return x,y

def get_load_vector(N1,N2):
    """
    :param Nlb: You can set the value of Nlb(number of local basis function), Nlb=3 here
    :param Nb: number of finite element node(unknowns)
    :param N: number of finite of element
    :return: load vector
    """
    Nlb=9
    nbn=4*(N1+N2)
    N=N1*N2
    Nb=(2*N1+1)*(2*N2+1)
    b=np.zeros((Nb,1),dtype=float)
    for n in range(N):
        for beta in range(Nlb):
            r=gauss_quadrature_load(N1,N2,n,beta)
            lis=Tb_function(N1,N2,n)
            b[lis[beta],0]+=r
    # Deal with the boundary nodes
    for k in range(nbn):
        i=boundaryedges(N1,N2,k)
        x,y=Pb_function(N1,N2,i)
        b[i,0]=Boundary_fun(x,y)
    return b
def gauss_quadrature_load(N1,N2,n_e,beta):
    Nlb=9
    X=np.array([0]*Nlb,dtype=float)
    Y=np.array([0]*Nlb,dtype=float)
    index_lis=Tb_function(N1,N2,n_e)
    p=Pb_function(N1,N2,index_lis[0])
    for i in range(Nlb):
        X[i],Y[i]=Pb_function(N1,N2,index_lis[i])
    h1=X[1]-X[0]
    h2=Y[3]-Y[0]
    #print("h1==",h1,"h2==",h2)
    if beta==0:
        r=(25/81)*f(-0.7745967,-0.7745967,h1,h2,p)*pusi0(-0.7745967,-0.7745967)+\
          (40/81)*f(-0.7745967,-0,h1,h2,p)*pusi0(-0.7745967,0)+\
          (25/81)*f(-0.7745967,0.7745967,h1,h2,p)*pusi0(-0.7745967,0.7745967)+\
          (40/81)*f(0,-0.7745967,h1,h2,p)*pusi0(0,-0.7745967)+\
          (64/81)*f(0,0,h1,h2,p)*pusi0(0,0)+\
          (40/81)*f(0,0.7745967,h1,h2,p)*pusi0(0,0.7745967)+\
          (25/81)*f(0.7745967,-0.7745967,h1,h2,p)*pusi0(0.7745967,-0.7745967)+\
          (40/81)*f(0.7745967,0,h1,h2,p)*pusi0(0.7745967,0)+\
          (25/81)*f(0.7745967,0.7745967,h1,h2,p)*pusi0(0.7745967,0.7745967)
        return r
    elif beta==1:
        r=(25/81)*f(-0.7745967,-0.7745967,h1,h2,p)*pusi1(-0.7745967,-0.7745967)+\
          (40/81)*f(-0.7745967,-0,h1,h2,p)*pusi1(-0.7745967,0)+\
          (25/81)*f(-0.7745967,0.7745967,h1,h2,p)*pusi1(-0.7745967,0.7745967)+\
          (40/81)*f(0,-0.7745967,h1,h2,p)*pusi1(0,-0.7745967)+\
          (64/81)*f(0,0,h1,h2,p)*pusi1(0,0)+\
          (40/81)*f(0,0.7745967,h1,h2,p)*pusi1(0,0.7745967)+\
          (25/81)*f(0.7745967,-0.7745967,h1,h2,p)*pusi1(0.7745967,-0.7745967)+\
          (40/81)*f(0.7745967,0,h1,h2,p)*pusi1(0.7745967,0)+\
          (25/81)*f(0.7745967,0.7745967,h1,h2,p)*pusi1(0.7745967,0.7745967)
        return r
    elif beta==2:
        r=(25/81)*f(-0.7745967,-0.7745967,h1,h2,p)*pusi2(-0.7745967,-0.7745967)+\
          (40/81)*f(-0.7745967,-0,h1,h2,p)*pusi2(-0.7745967,0)+\
          (25/81)*f(-0.7745967,0.7745967,h1,h2,p)*pusi2(-0.7745967,0.7745967)+\
          (40/81)*f(0,-0.7745967,h1,h2,p)*pusi2(0,-0.7745967)+\
          (64/81)*f(0,0,h1,h2,p)*pusi2(0,0)+\
          (40/81)*f(0,0.7745967,h1,h2,p)*pusi2(0,0.7745967)+\
          (25/81)*f(0.7745967,-0.7745967,h1,h2,p)*pusi2(0.7745967,-0.7745967)+\
          (40/81)*f(0.7745967,0,h1,h2,p)*pusi2(0.7745967,0)+\
          (25/81)*f(0.7745967,0.7745967,h1,h2,p)*pusi2(0.7745967,0.7745967)
        return r
    elif beta==3:
        r=(25/81)*f(-0.7745967,-0.7745967,h1,h2,p)*pusi3(-0.7745967,-0.7745967)+\
          (40/81)*f(-0.7745967,-0,h1,h2,p)*pusi3(-0.7745967,0)+\
          (25/81)*f(-0.7745967,0.7745967,h1,h2,p)*pusi3(-0.7745967,0.7745967)+\
          (40/81)*f(0,-0.7745967,h1,h2,p)*pusi3(0,-0.7745967)+\
          (64/81)*f(0,0,h1,h2,p)*pusi3(0,0)+\
          (40/81)*f(0,0.7745967,h1,h2,p)*pusi3(0,0.7745967)+\
          (25/81)*f(0.7745967,-0.7745967,h1,h2,p)*pusi3(0.7745967,-0.7745967)+\
          (40/81)*f(0.7745967,0,h1,h2,p)*pusi3(0.7745967,0)+\
          (25/81)*f(0.7745967,0.7745967,h1,h2,p)*pusi3(0.7745967,0.7745967)
        return r
    elif beta==4:
        r=(25/81)*f(-0.7745967,-0.7745967,h1,h2,p)*pusi4(-0.7745967,-0.7745967)+\
          (40/81)*f(-0.7745967,-0,h1,h2,p)*pusi4(-0.7745967,0)+\
          (25/81)*f(-0.7745967,0.7745967,h1,h2,p)*pusi4(-0.7745967,0.7745967)+\
          (40/81)*f(0,-0.7745967,h1,h2,p)*pusi4(0,-0.7745967)+\
          (64/81)*f(0,0,h1,h2,p)*pusi4(0,0)+\
          (40/81)*f(0,0.7745967,h1,h2,p)*pusi4(0,0.7745967)+\
          (25/81)*f(0.7745967,-0.7745967,h1,h2,p)*pusi4(0.7745967,-0.7745967)+\
          (40/81)*f(0.7745967,0,h1,h2,p)*pusi4(0.7745967,0)+\
          (25/81)*f(0.7745967,0.7745967,h1,h2,p)*pusi4(0.7745967,0.7745967)
        return r
    elif beta==5:
        r=(25/81)*f(-0.7745967,-0.7745967,h1,h2,p)*pusi5(-0.7745967,-0.7745967)+\
          (40/81)*f(-0.7745967,-0,h1,h2,p)*pusi5(-0.7745967,0)+\
          (25/81)*f(-0.7745967,0.7745967,h1,h2,p)*pusi5(-0.7745967,0.7745967)+\
          (40/81)*f(0,-0.7745967,h1,h2,p)*pusi5(0,-0.7745967)+\
          (64/81)*f(0,0,h1,h2,p)*pusi5(0,0)+\
          (40/81)*f(0,0.7745967,h1,h2,p)*pusi5(0,0.7745967)+\
          (25/81)*f(0.7745967,-0.7745967,h1,h2,p)*pusi5(0.7745967,-0.7745967)+\
          (40/81)*f(0.7745967,0,h1,h2,p)*pusi5(0.7745967,0)+\
          (25/81)*f(0.7745967,0.7745967,h1,h2,p)*pusi5(0.7745967,0.7745967)
        return r
    elif beta==6:
        r=(25/81)*f(-0.7745967,-0.7745967,h1,h2,p)*pusi6(-0.7745967,-0.7745967)+\
          (40/81)*f(-0.7745967,-0,h1,h2,p)*pusi6(-0.7745967,0)+\
          (25/81)*f(-0.7745967,0.7745967,h1,h2,p)*pusi6(-0.7745967,0.7745967)+\
          (40/81)*f(0,-0.7745967,h1,h2,p)*pusi6(0,-0.7745967)+\
          (64/81)*f(0,0,h1,h2,p)*pusi6(0,0)+\
          (40/81)*f(0,0.7745967,h1,h2,p)*pusi6(0,0.7745967)+\
          (25/81)*f(0.7745967,-0.7745967,h1,h2,p)*pusi6(0.7745967,-0.7745967)+\
          (40/81)*f(0.7745967,0,h1,h2,p)*pusi6(0.7745967,0)+\
          (25/81)*f(0.7745967,0.7745967,h1,h2,p)*pusi6(0.7745967,0.7745967)
        return r
    elif beta==7:
        r=(25/81)*f(-0.7745967,-0.7745967,h1,h2,p)*pusi7(-0.7745967,-0.7745967)+\
          (40/81)*f(-0.7745967,-0,h1,h2,p)*pusi7(-0.7745967,0)+\
          (25/81)*f(-0.7745967,0.7745967,h1,h2,p)*pusi7(-0.7745967,0.7745967)+\
          (40/81)*f(0,-0.7745967,h1,h2,p)*pusi7(0,-0.7745967)+\
          (64/81)*f(0,0,h1,h2,p)*pusi7(0,0)+\
          (40/81)*f(0,0.7745967,h1,h2,p)*pusi7(0,0.7745967)+\
          (25/81)*f(0.7745967,-0.7745967,h1,h2,p)*pusi7(0.7745967,-0.7745967)+\
          (40/81)*f(0.7745967,0,h1,h2,p)*pusi7(0.7745967,0)+\
          (25/81)*f(0.7745967,0.7745967,h1,h2,p)*pusi7(0.7745967,0.7745967)
        return r
    elif beta==8:
        r=(25/81)*f(-0.7745967,-0.7745967,h1,h2,p)*pusi8(-0.7745967,-0.7745967)+\
          (40/81)*f(-0.7745967,-0,h1,h2,p)*pusi8(-0.7745967,0)+\
          (25/81)*f(-0.7745967,0.7745967,h1,h2,p)*pusi8(-0.7745967,0.7745967)+\
          (40/81)*f(0,-0.7745967,h1,h2,p)*pusi8(0,-0.7745967)+\
          (64/81)*f(0,0,h1,h2,p)*pusi8(0,0)+\
          (40/81)*f(0,0.7745967,h1,h2,p)*pusi8(0,0.7745967)+\
          (25/81)*f(0.7745967,-0.7745967,h1,h2,p)*pusi8(0.7745967,-0.7745967)+\
          (40/81)*f(0.7745967,0,h1,h2,p)*pusi8(0.7745967,0)+\
          (25/81)*f(0.7745967,0.7745967,h1,h2,p)*pusi8(0.7745967,0.7745967)
        return r

def Boundary_fun(x,y):
    """ adding boundary condition! """
    if x==-1:
        return -1.5*y*(1-y)*np.exp(-1+y)
    elif x==1:
        return 0.5*y*(1-y)*np.exp(1+y)
    elif y==-1:
        return -2*x*(1-0.5*x)*np.exp(x-1)
    elif y==1:
        return 0

def f(x,y,h1,h2,p):
    xlo=0.5*h1*(x+1)+p[0]
    ylo=0.5*h2*(y+1)+p[1]
    return -ylo*(1-ylo)*(1-xlo-0.5*xlo**2)*np.exp(xlo+ylo)-xlo*(1-0.5*xlo)*(-3*ylo-ylo**2)*np.exp(xlo+ylo)

def pusi0(x,y):
    return 0.25*x*y-0.25*x**2*y-0.25*x*y**2+0.25*x**2*y**2
def pusi1(xlo,ylo):
    return -0.25*xlo*ylo-0.25*xlo**2*ylo+0.25*xlo*ylo**2+0.25*xlo**2*ylo**2
def pusi2(xlo,ylo):
    return 0.25*xlo*ylo+0.25*xlo**2*ylo+0.25*xlo*ylo**2+0.25*xlo**2*ylo**2
def pusi3(xlo,ylo):
    return -0.25*xlo*ylo+0.25*xlo**2*ylo-0.25*xlo*ylo**2+0.25*xlo**2*ylo**2
def pusi4(xlo,ylo):
    return -0.5*ylo+0.5*ylo**2+0.5*xlo**2*ylo-0.5*xlo**2*ylo**2
def pusi5(xlo,ylo):
    return 0.5*xlo+0.5*xlo**2-0.5*xlo*ylo**2-0.5*xlo**2*ylo**2
def pusi6(xlo,ylo):
    return 0.5*ylo+0.5*ylo**2-0.5*xlo**2*ylo-0.5*xlo**2*ylo**2
def pusi7(xlo,ylo):
    return -0.5*xlo+0.5*xlo**2+0.5*xlo*ylo**2-0.5*xlo**2*ylo**2
def pusi8(xlo,ylo):
    return 1-xlo**2-ylo**2+xlo**2*ylo**2

def boundaryedges(N1,N2,k):
    right1=[]
    top1=[]
    Nb=(2*N1+1)*(2*N2+1)
    nbn=4*(N1+N2)
    boundaryNode=np.zeros((2,nbn),dtype=int)
    boundaryNode[0,:]=[-1]*nbn
    mat=np.arange(Nb).reshape((2*N1+1,2*N2+1))
    top=list(mat[0,:])
    bottom=list(mat[2*N1,:])
    left=list(mat[:,0])
    right=list(mat[:,2*N2])
    for i in range(len(right)):
        right1.append(right.pop())
    for j in range(len(top)):
        top1.append(top.pop())
    del bottom[0]
    del right1[0]
    del top1[0]
    del top1[-1]
    index=left+bottom+right1+top1
    boundaryNode[1,:]=index
    #print("The boundaryNode mat is: ",'\n',boundaryNode)
    return boundaryNode[1,k]

def U_analy(x,y):
    return x*y*(1-0.5*x)*(1-y)*np.exp(x+y)
def U_loc(N1,N2,n_e):
    Nlb=9
    x=np.array([0]*Nlb,dtype=float)
    y=np.array([0]*Nlb,dtype=float)
    index_lis=Tb_function(N1,N2,n_e)
    for i in range(Nlb):
        x[i],y[i]=Pb_function(N1,N2,index_lis[i])
    x0=0.5*(x[0]+x[1])
    y0=0.5*(y[0]+y[3])
    result=U_analy(x0,y0)
    return result
def Wn(N1,N2,u,n_e):
    """
    Wn is the approximation function of U_analy
    :param u: numerical solution computed by A and b
    :param beta: order of basis function
    :param n_e: order of finite elements
    :return: approximation value of U_analy
    """
    Nlb=9
    x=np.array([0]*Nlb,dtype=float)
    y=np.array([0]*Nlb,dtype=float)
    #x=np.array([-0.7745967,-0.7745967,-0.7745967,0,0,0,0.7745967,0.7745967,0.7745967])
    #y=np.array([-0.7745967,0,0.7745967]*3)
    ub=[0]*Nlb
    index_lis=Tb_function(N1,N2,n_e)
    for i in range(Nlb):
        x[i],y[i]=Pb_function(N1,N2,index_lis[i])
        ub[i]=u[int(index_lis[i])]
    h1=x[1]-x[0]
    h2=y[3]-y[0]
    #print("h1 is: ",h1,"h2 is: ",h2)
    xm=0.5*(x[0]+x[1])
    ym=0.5*(y[0]+y[3])
    x_t=(1/h1)*(2*xm-2*x[0]-h1)
    y_t=(1/h2)*(2*ym-2*y[0]-h2)
    W=ub[0]*pusi0(x_t,y_t)+ub[1]*pusi1(x_t,y_t)+ub[2]*pusi2(x_t,y_t)+ub[3]*pusi3(x_t,y_t)+ub[4]*pusi4(x_t,y_t)+\
      ub[5]*pusi5(x_t,y_t)+ub[6]*pusi6(x_t,y_t)+ub[7]*pusi7(x_t,y_t)+ub[8]*pusi8(x_t,y_t)
    return W
def get_err(N1,N2,u):
    N=N1*N2
    err=[]
    for i in range(N):
        Ua=U_loc(N1,N2,i)
        Wnu=Wn(N1,N2,u,i)
        err.append(abs(Ua-Wnu))
    #print("The err list: ",err)
    error=max(err)
    return error

i=8
N1=i
N2=i
A=get_stiffness_mat(N1,N2)
# print("Det(A)= ",np.linalg.det(A))
#print("A is:",'\n',A)
b=get_load_vector(N1,N2)
# print("The b is: ",'\n',b)
inv_A=np.linalg.inv(A)
u=np.dot(inv_A,b)
print("u is: ",'\n',u)
err=get_err(N1,N2,u)
print("The err is: ",err)
