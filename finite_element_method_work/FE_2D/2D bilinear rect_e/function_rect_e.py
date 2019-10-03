import numpy as np

def get_stiffness_mat(N1,N2):
    """
    :param Nlb: You can set the value of Nlb(number of local basis function), Nlb=4 here
    :param Nm: number of unknowns, or of finite element nodes
    :param N: number of finite elements
    :return: stiffness matrix
    """
    Nb=(N1+1)*(N2+1)
    nbn=2*(N1+N2)
    N=N1*N2
    A=np.zeros((Nb,Nb),dtype=float)           #TODO: taking advantage of sparse matrix may accelerate computing
    for n in range(N):
        for alpha in range(4):
            for beta in range(4):
                r=gauss_quadrature_stiff(N1,N2,n,alpha,beta)
                lis=T_function(N1,N2,n)
                A[lis[beta],lis[alpha]]+=r
    # Deal with the boundary nodes
    for k in range(nbn):
        i=boundaryedges(N1,N2,k)
        A[i,:]=0
        A[i,i]=1
    return A

def boundaryedges(N1,N2,k):
    index=[]
    right1=[]
    top1=[]
    Nb=(N1+1)*(N2+1)
    nbn=2*(N1+N2)
    boundaryNode=np.zeros((2,nbn),dtype=int)
    boundaryNode[0,:]=[-1]*nbn
    array=np.arange(Nb)
    mat=array.reshape((N1+1,N2+1))
    top=list(mat[0,:])
    bottom=list(mat[N1,:])
    left=list(mat[:,0])
    right=list(mat[:,N2])
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

def gauss_quadrature_stiff(N1,N2,n,alpha,beta):
    Nlb=4
    X=np.array([0]*Nlb,dtype=float)
    Y=np.array([0]*Nlb,dtype=float)
    index_lis=T_function(N1,N2,n)
    for i in range(Nlb):
        X[i],Y[i]=P_function(N1,N2,index_lis[i])
    h1=X[1]-X[0]
    #h2=Y[3]-Y[0]             #TODO: set h1==h2 by default
    coef=1.0/(4*(h1**2))
    if alpha==0 and beta==0:
        r=coef*((25/81)*phi0x(-0.7745967,-0.7745967)*phi0x(-0.7745967,-0.7745967)+
                (40/81)*phi0x(-0.7745967,0)*phi0x(-0.7745967,0)+
                (25/81)*phi0x(-0.7745967,0.7745967)*phi0x(-0.7745967,0.7745967)+
                (40/81)*phi0x(0,-0.7745967)*phi0x(0,-0.7745967)+
                (64/81)*phi0x(0,0)*phi0x(0,0)+
                (40/81)*phi0x(0,0.7745967)*phi0x(0,0.7745967)+
                (25/81)*phi0x(0.7745967,-0.7745967)*phi0x(0.7745967,-0.7745967)+
                (40/81)*phi0x(0.7745967,0)*phi0x(0.7745967,0)+
                (25/81)*phi0x(0.7745967,0.7745967)*phi0x(0.7745967,0.7745967)  )*2
    elif (alpha==1 and beta==1):
        r=coef*((25/81)*phi1x(-0.7745967,-0.7745967)*phi1x(-0.7745967,-0.7745967)+
                (40/81)*phi1x(-0.7745967,0)*phi1x(-0.7745967,0)+
                (25/81)*phi1x(-0.7745967,0.7745967)*phi1x(-0.7745967,0.7745967)+
                (40/81)*phi1x(0,-0.7745967)*phi1x(0,-0.7745967)+
                (64/81)*phi1x(0,0)*phi1x(0,0)+
                (40/81)*phi1x(0,0.7745967)*phi1x(0,0.7745967)+
                (25/81)*phi1x(0.7745967,-0.7745967)*phi1x(0.7745967,-0.7745967)+
                (40/81)*phi1x(0.7745967,0)*phi1x(0.7745967,0)+
                (25/81)*phi1x(0.7745967,0.7745967)*phi1x(0.7745967,0.7745967)  )*2
    elif (alpha==2 and beta==2):
        r=coef*((25/81)*phi2x(-0.7745967,-0.7745967)*phi2x(-0.7745967,-0.7745967)+
                (40/81)*phi2x(-0.7745967,0)*phi2x(-0.7745967,0)+
                (25/81)*phi2x(-0.7745967,0.7745967)*phi2x(-0.7745967,0.7745967)+
                (40/81)*phi2x(0,-0.7745967)*phi2x(0,-0.7745967)+
                (64/81)*phi2x(0,0)*phi2x(0,0)+
                (40/81)*phi2x(0,0.7745967)*phi2x(0,0.7745967)+
                (25/81)*phi2x(0.7745967,-0.7745967)*phi2x(0.7745967,-0.7745967)+
                (40/81)*phi2x(0.7745967,0)*phi2x(0.7745967,0)+
                (25/81)*phi2x(0.7745967,0.7745967)*phi2x(0.7745967,0.7745967)  )*2
    elif (alpha==3 and beta==3):
        r=coef*((25/81)*phi3x(-0.7745967,-0.7745967)*phi3x(-0.7745967,-0.7745967)+
                (40/81)*phi3x(-0.7745967,0)*phi3x(-0.7745967,0)+
                (25/81)*phi3x(-0.7745967,0.7745967)*phi3x(-0.7745967,0.7745967)+
                (40/81)*phi3x(0,-0.7745967)*phi3x(0,-0.7745967)+
                (64/81)*phi3x(0,0)*phi3x(0,0)+
                (40/81)*phi3x(0,0.7745967)*phi3x(0,0.7745967)+
                (25/81)*phi3x(0.7745967,-0.7745967)*phi3x(0.7745967,-0.7745967)+
                (40/81)*phi3x(0.7745967,0)*phi3x(0.7745967,0)+
                (25/81)*phi3x(0.7745967,0.7745967)*phi3x(0.7745967,0.7745967)  )*2
    elif (alpha==0 and beta==1) or (alpha==1 and beta==0):
        r=coef*((25/81)*phi0x(-0.7745967,-0.7745967)*phi1x(-0.7745967,-0.7745967)+
                (40/81)*phi0x(-0.7745967,0)*phi1x(-0.7745967,0)+
                (25/81)*phi0x(-0.7745967,0.7745967)*phi1x(-0.7745967,0.7745967)+
                (40/81)*phi0x(0,-0.7745967)*phi1x(0,-0.7745967)+
                (64/81)*phi0x(0,0)*phi1x(0,0)+
                (40/81)*phi0x(0,0.7745967)*phi1x(0,0.7745967)+
                (25/81)*phi0x(0.7745967,-0.7745967)*phi1x(0.7745967,-0.7745967)+
                (40/81)*phi0x(0.7745967,0)*phi1x(0.7745967,0)+
                (25/81)*phi0x(0.7745967,0.7745967)*phi1x(0.7745967,0.7745967)+

                (25/81)*phi0y(-0.7745967,-0.7745967)*phi1y(-0.7745967,-0.7745967)+
                (40/81)*phi0y(-0.7745967,0)*phi1y(-0.7745967,0)+
                (25/81)*phi0y(-0.7745967,0.7745967)*phi1y(-0.7745967,0.7745967)+
                (40/81)*phi0y(0,-0.7745967)*phi1y(0,-0.7745967)+
                (64/81)*phi0y(0,0)*phi1y(0,0)+
                (40/81)*phi0y(0,0.7745967)*phi1y(0,0.7745967)+
                (25/81)*phi0y(0.7745967,-0.7745967)*phi1y(0.7745967,-0.7745967)+
                (40/81)*phi0y(0.7745967,0)*phi1y(0.7745967,0)+
                (25/81)*phi0y(0.7745967,0.7745967)*phi1y(0.7745967,0.7745967))
    elif (alpha==0 and beta==2) or (alpha==2 and beta==0):
        r=coef*((25/81)*phi0x(-0.7745967,-0.7745967)*phi2x(-0.7745967,-0.7745967)+
                (40/81)*phi0x(-0.7745967,0)*phi2x(-0.7745967,0)+
                (25/81)*phi0x(-0.7745967,0.7745967)*phi2x(-0.7745967,0.7745967)+
                (40/81)*phi0x(0,-0.7745967)*phi2x(0,-0.7745967)+
                (64/81)*phi0x(0,0)*phi2x(0,0)+
                (40/81)*phi0x(0,0.7745967)*phi2x(0,0.7745967)+
                (25/81)*phi0x(0.7745967,-0.7745967)*phi2x(0.7745967,-0.7745967)+
                (40/81)*phi0x(0.7745967,0)*phi2x(0.7745967,0)+
                (25/81)*phi0x(0.7745967,0.7745967)*phi2x(0.7745967,0.7745967)+

                (25/81)*phi0y(-0.7745967,-0.7745967)*phi2y(-0.7745967,-0.7745967)+
                (40/81)*phi0y(-0.7745967,0)*phi2y(-0.7745967,0)+
                (25/81)*phi0y(-0.7745967,0.7745967)*phi2y(-0.7745967,0.7745967)+
                (40/81)*phi0y(0,-0.7745967)*phi2y(0,-0.7745967)+
                (64/81)*phi0y(0,0)*phi2y(0,0)+
                (40/81)*phi0y(0,0.7745967)*phi2y(0,0.7745967)+
                (25/81)*phi0y(0.7745967,-0.7745967)*phi2y(0.7745967,-0.7745967)+
                (40/81)*phi0y(0.7745967,0)*phi2y(0.7745967,0)+
                (25/81)*phi0y(0.7745967,0.7745967)*phi2y(0.7745967,0.7745967))
    elif (alpha==0 and beta==3) or (alpha==3 and beta==0):
        r=coef*((25/81)*phi0x(-0.7745967,-0.7745967)*phi3x(-0.7745967,-0.7745967)+
                (40/81)*phi0x(-0.7745967,0)*phi3x(-0.7745967,0)+
                (25/81)*phi0x(-0.7745967,0.7745967)*phi3x(-0.7745967,0.7745967)+
                (40/81)*phi0x(0,-0.7745967)*phi3x(0,-0.7745967)+
                (64/81)*phi0x(0,0)*phi3x(0,0)+
                (40/81)*phi0x(0,0.7745967)*phi3x(0,0.7745967)+
                (25/81)*phi0x(0.7745967,-0.7745967)*phi3x(0.7745967,-0.7745967)+
                (40/81)*phi0x(0.7745967,0)*phi3x(0.7745967,0)+
                (25/81)*phi0x(0.7745967,0.7745967)*phi3x(0.7745967,0.7745967)+

                (25/81)*phi0y(-0.7745967,-0.7745967)*phi3y(-0.7745967,-0.7745967)+
                (40/81)*phi0y(-0.7745967,0)*phi3y(-0.7745967,0)+
                (25/81)*phi0y(-0.7745967,0.7745967)*phi3y(-0.7745967,0.7745967)+
                (40/81)*phi0y(0,-0.7745967)*phi3y(0,-0.7745967)+
                (64/81)*phi0y(0,0)*phi3y(0,0)+
                (40/81)*phi0y(0,0.7745967)*phi3y(0,0.7745967)+
                (25/81)*phi0y(0.7745967,-0.7745967)*phi3y(0.7745967,-0.7745967)+
                (40/81)*phi0y(0.7745967,0)*phi3y(0.7745967,0)+
                (25/81)*phi0y(0.7745967,0.7745967)*phi3y(0.7745967,0.7745967))
    elif (alpha==1 and beta==2) or (alpha==2 and beta==1):
        r=coef*((25/81)*phi1x(-0.7745967,-0.7745967)*phi2x(-0.7745967,-0.7745967)+
                (40/81)*phi1x(-0.7745967,0)*phi2x(-0.7745967,0)+
                (25/81)*phi1x(-0.7745967,0.7745967)*phi2x(-0.7745967,0.7745967)+
                (40/81)*phi1x(0,-0.7745967)*phi2x(0,-0.7745967)+
                (64/81)*phi1x(0,0)*phi2x(0,0)+
                (40/81)*phi1x(0,0.7745967)*phi2x(0,0.7745967)+
                (25/81)*phi1x(0.7745967,-0.7745967)*phi2x(0.7745967,-0.7745967)+
                (40/81)*phi1x(0.7745967,0)*phi2x(0.7745967,0)+
                (25/81)*phi1x(0.7745967,0.7745967)*phi2x(0.7745967,0.7745967)+

                (25/81)*phi1y(-0.7745967,-0.7745967)*phi2y(-0.7745967,-0.7745967)+
                (40/81)*phi1y(-0.7745967,0)*phi2y(-0.7745967,0)+
                (25/81)*phi1y(-0.7745967,0.7745967)*phi2y(-0.7745967,0.7745967)+
                (40/81)*phi1y(0,-0.7745967)*phi2y(0,-0.7745967)+
                (64/81)*phi1y(0,0)*phi2y(0,0)+
                (40/81)*phi1y(0,0.7745967)*phi2y(0,0.7745967)+
                (25/81)*phi1y(0.7745967,-0.7745967)*phi2y(0.7745967,-0.7745967)+
                (40/81)*phi1y(0.7745967,0)*phi2y(0.7745967,0)+
                (25/81)*phi1y(0.7745967,0.7745967)*phi2y(0.7745967,0.7745967))
    elif (alpha==1 and beta==3) or (alpha==3 and beta==1):
        r=coef*((25/81)*phi1x(-0.7745967,-0.7745967)*phi3x(-0.7745967,-0.7745967)+
                (40/81)*phi1x(-0.7745967,0)*phi3x(-0.7745967,0)+
                (25/81)*phi1x(-0.7745967,0.7745967)*phi3x(-0.7745967,0.7745967)+
                (40/81)*phi1x(0,-0.7745967)*phi3x(0,-0.7745967)+
                (64/81)*phi1x(0,0)*phi3x(0,0)+
                (40/81)*phi1x(0,0.7745967)*phi3x(0,0.7745967)+
                (25/81)*phi1x(0.7745967,-0.7745967)*phi3x(0.7745967,-0.7745967)+
                (40/81)*phi1x(0.7745967,0)*phi3x(0.7745967,0)+
                (25/81)*phi1x(0.7745967,0.7745967)*phi3x(0.7745967,0.7745967)+

                (25/81)*phi1y(-0.7745967,-0.7745967)*phi3y(-0.7745967,-0.7745967)+
                (40/81)*phi1y(-0.7745967,0)*phi3y(-0.7745967,0)+
                (25/81)*phi1y(-0.7745967,0.7745967)*phi3y(-0.7745967,0.7745967)+
                (40/81)*phi1y(0,-0.7745967)*phi3y(0,-0.7745967)+
                (64/81)*phi1y(0,0)*phi3y(0,0)+
                (40/81)*phi1y(0,0.7745967)*phi3y(0,0.7745967)+
                (25/81)*phi1y(0.7745967,-0.7745967)*phi3y(0.7745967,-0.7745967)+
                (40/81)*phi1y(0.7745967,0)*phi3y(0.7745967,0)+
                (25/81)*phi1y(0.7745967,0.7745967)*phi3y(0.7745967,0.7745967))
    elif (alpha==2 and beta==3) or (alpha==3 and beta==2):
        r=coef*((25/81)*phi3x(-0.7745967,-0.7745967)*phi2x(-0.7745967,-0.7745967)+
                (40/81)*phi3x(-0.7745967,0)*phi2x(-0.7745967,0)+
                (25/81)*phi3x(-0.7745967,0.7745967)*phi2x(-0.7745967,0.7745967)+
                (40/81)*phi3x(0,-0.7745967)*phi2x(0,-0.7745967)+
                (64/81)*phi3x(0,0)*phi2x(0,0)+
                (40/81)*phi3x(0,0.7745967)*phi2x(0,0.7745967)+
                (25/81)*phi3x(0.7745967,-0.7745967)*phi2x(0.7745967,-0.7745967)+
                (40/81)*phi3x(0.7745967,0)*phi2x(0.7745967,0)+
                (25/81)*phi3x(0.7745967,0.7745967)*phi2x(0.7745967,0.7745967)+

                (25/81)*phi3y(-0.7745967,-0.7745967)*phi2y(-0.7745967,-0.7745967)+
                (40/81)*phi3y(-0.7745967,0)*phi2y(-0.7745967,0)+
                (25/81)*phi3y(-0.7745967,0.7745967)*phi2y(-0.7745967,0.7745967)+
                (40/81)*phi3y(0,-0.7745967)*phi2y(0,-0.7745967)+
                (64/81)*phi3y(0,0)*phi2y(0,0)+
                (40/81)*phi3y(0,0.7745967)*phi2y(0,0.7745967)+
                (25/81)*phi3y(0.7745967,-0.7745967)*phi2y(0.7745967,-0.7745967)+
                (40/81)*phi3y(0.7745967,0)*phi2y(0.7745967,0)+
                (25/81)*phi3y(0.7745967,0.7745967)*phi2y(0.7745967,0.7745967))
    return r

def T_function(N1,N2,n_e):
    """
    set Nlb==4, so there are 4 rows of T matrix
    :param N1: number of elements of x-axis direction
    :param N2: number of elements of y-axis direction
    :param n_e: order of finite element (n-th element)
    :return: global node index of n-th element
    """
    N=N1*N2
    T=np.zeros((4,N),dtype=int)
    T[:,0]=[0,N2+1,N2+2,1]
    for k in range(1,N2):
        T[:,k]=T[:,k-1]+[1,1,1,1]
    for i in range(1,N1):
        T[:,N2*i]=T[:,N2*i-1]+[2,2,2,2]
        for j in range(1,N2):
            T[:,N2*i+j]=T[:,N2*i+j-1]+[1,1,1,1]
    #print("T is: ",'\n',T)
    return T[:,n_e]

def P_function(N1,N2,n_g):
    Nb=(N1+1)*(N2+1)
    left=-1
    right=1
    bottom=-1
    top=1
    lis=[]
    P=np.zeros((2,Nb),dtype=float)
    x_array=np.linspace(left,right,N1+1,dtype=float)
    y_array=np.linspace(bottom,top,N2+1,dtype=float)
    x_lis=list(x_array)
    for i in x_lis:
        lis+=[i]*(N2+1)
    P[0,:]=lis
    P[1,:]=list(y_array)*(N1+1)
    #print("P is: ",'\n',P)
    x,y=P[:,n_g]
    return x,y

def phi0x(x,y):
    return (-1+y)
def phi1x(x,y):
    return (1-y)
def phi2x(x,y):
    return (1+y)
def phi3x(x,y):
    return (-1-y)
def phi0y(x,y):
    return (-1+x)
def phi1y(x,y):
    return (-1-x)
def phi2y(x,y):
    return (1+x)
def phi3y(x,y):
    return (1-x)

def get_load_vector(N1,N2):
    """
    :param Nlb: You can set the value of Nlb(number of local basis function), Nlb=3 here
    :param Nb: number of finite element node(unknowns)
    :param N: number of finite of element
    :return: load vector
    """
    Nlb=4
    nbn=2*(N1+N2)
    N=N1*N2
    Nb=(N1+1)*(N2+1)
    b=np.zeros((Nb,1),dtype=float)
    for n in range(N):
        for beta in range(Nlb):
            r=gauss_quadrature_load(N1,N2,n,beta)
            lis=T_function(N1,N2,n)
            b[lis[beta],0]+=r
    # Deal with the boundary nodes
    for k in range(nbn):
        i=boundaryedges(N1,N2,k)
        x,y=P_function(N1,N2,i)
        b[i,0]=Boundary_fun(x,y)
    return b

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

def gauss_quadrature_load(N1,N2,n_e,beta):
    Nlb=4
    X=np.array([0]*Nlb,dtype=float)
    Y=np.array([0]*Nlb,dtype=float)
    index_lis=T_function(N1,N2,n_e)
    for i in range(Nlb):
        X[i],Y[i]=P_function(N1,N2,index_lis[i])
    h1=X[1]-X[0]
    h2=Y[3]-Y[0]
    #print("h1==",h1,"h2==",h2)
    if beta==0:
        r=(25/81)*(-affin_y(-0.7745967,Y[0],h2)*(1-affin_y(-0.7745967,Y[0],h2))*
                   (1-affin_x(-0.7745967,X[0],h1)-0.5*affin_x(-0.7745967,X[0],h1)*affin_x(-0.7745967,X[0],h1))*
                   np.exp(affin_x(-0.7745967,X[0],h1)+affin_y(-0.7745967,Y[0],h2))-affin_x(-0.7745967,X[0],h1)*
                   (1-0.5*affin_x(-0.7745967,X[0],h1))*(-3*affin_y(-0.7745967,Y[0],h2)-affin_y(-0.7745967,Y[0],h2)*affin_y(-0.7745967,Y[0],h2))
                   *np.exp(affin_x(-0.7745967,X[0],h1)+affin_y(-0.7745967,Y[0],h2)))*pus0(-0.7745967,-0.7745967)+\
          (40/81)*(-affin_y(0,Y[0],h2)*(1-affin_y(-0,Y[0],h2))*
                   (1-affin_x(-0.7745967,X[0],h1)-0.5*affin_x(-0.7745967,X[0],h1)*affin_x(-0.7745967,X[0],h1))*
                   np.exp(affin_x(-0.7745967,X[0],h1)+affin_y(0,Y[0],h2))-affin_x(-0.7745967,X[0],h1)*
                   (1-0.5*affin_x(-0.7745967,X[0],h1))*(-3*affin_y(0,Y[0],h2)-affin_y(0,Y[0],h2)*affin_y(0,Y[0],h2))
                   *np.exp(affin_x(-0.7745967,X[0],h1)+affin_y(0,Y[0],h2)))*pus0(-0.7745967,0)+\
          (25/81)*(-affin_y(0.7745967,Y[0],h2)*(1-affin_y(0.7745967,Y[0],h2))*
                   (1-affin_x(-0.7745967,X[0],h1)-0.5*affin_x(-0.7745967,X[0],h1)*affin_x(-0.7745967,X[0],h1))*
                   np.exp(affin_x(-0.7745967,X[0],h1)+affin_y(0.7745967,Y[0],h2))-affin_x(-0.7745967,X[0],h1)*
                   (1-0.5*affin_x(-0.7745967,X[0],h1))*(-3*affin_y(0.7745967,Y[0],h2)-affin_y(0.7745967,Y[0],h2)*affin_y(0.7745967,Y[0],h2))
                   *np.exp(affin_x(-0.7745967,X[0],h1)+affin_y(0.7745967,Y[0],h2)))*pus0(-0.7745967,0.7745967)+\
          (40/81)*(-affin_y(-0.7745967,Y[0],h2)*(1-affin_y(-0.7745967,Y[0],h2))*
                   (1-affin_x(0,X[0],h1)-0.5*affin_x(0,X[0],h1)*affin_x(0,X[0],h1))*
                   np.exp(affin_x(0,X[0],h1)+affin_y(-0.7745967,Y[0],h2))-affin_x(0,X[0],h1)*
                   (1-0.5*affin_x(0,X[0],h1))*(-3*affin_y(-0.7745967,Y[0],h2)-affin_y(-0.7745967,Y[0],h2)*affin_y(-0.7745967,Y[0],h2))
                   *np.exp(affin_x(0,X[0],h1)+affin_y(-0.7745967,Y[0],h2)))*pus0(0,-0.7745967)+\
          (64/81)*(-affin_y(0,Y[0],h2)*(1-affin_y(0,Y[0],h2))*
                   (1-affin_x(0,X[0],h1)-0.5*affin_x(0,X[0],h1)*affin_x(0,X[0],h1))*
                   np.exp(affin_x(0,X[0],h1)+affin_y(0,Y[0],h2))-affin_x(0,X[0],h1)*
                   (1-0.5*affin_x(0,X[0],h1))*(-3*affin_y(0,Y[0],h2)-affin_y(0,Y[0],h2)*affin_y(0,Y[0],h2))
                   *np.exp(affin_x(0,X[0],h1)+affin_y(0,Y[0],h2)))*pus0(0,0)+\
          (40/81)*(-affin_y(0.7745967,Y[0],h2)*(1-affin_y(0.7745967,Y[0],h2))*
                   (1-affin_x(0,X[0],h1)-0.5*affin_x(0,X[0],h1)*affin_x(0,X[0],h1))*
                   np.exp(affin_x(0,X[0],h1)+affin_y(0.7745967,Y[0],h2))-affin_x(0,X[0],h1)*
                   (1-0.5*affin_x(0,X[0],h1))*(-3*affin_y(0.7745967,Y[0],h2)-affin_y(0.7745967,Y[0],h2)*affin_y(0.7745967,Y[0],h2))
                   *np.exp(affin_x(0,X[0],h1)+affin_y(0.7745967,Y[0],h2)))*pus0(0,0.7745967)+\
          (25/81)*(-affin_y(-0.7745967,Y[0],h2)*(1-affin_y(-0.7745967,Y[0],h2))*
                   (1-affin_x(0.7745967,X[0],h1)-0.5*affin_x(0.7745967,X[0],h1)*affin_x(0.7745967,X[0],h1))*
                   np.exp(affin_x(0.7745967,X[0],h1)+affin_y(-0.7745967,Y[0],h2))-affin_x(0.7745967,X[0],h1)*
                   (1-0.5*affin_x(0.7745967,X[0],h1))*(-3*affin_y(-0.7745967,Y[0],h2)-affin_y(-0.7745967,Y[0],h2)*affin_y(-0.7745967,Y[0],h2))
                   *np.exp(affin_x(0.7745967,X[0],h1)+affin_y(-0.7745967,Y[0],h2)))*pus0(0.7745967,-0.7745967)+\
          (40/81)*(-affin_y(0,Y[0],h2)*(1-affin_y(-0,Y[0],h2))*
                   (1-affin_x(0.7745967,X[0],h1)-0.5*affin_x(0.7745967,X[0],h1)*affin_x(0.7745967,X[0],h1))*
                   np.exp(affin_x(0.7745967,X[0],h1)+affin_y(0,Y[0],h2))-affin_x(0.7745967,X[0],h1)*
                   (1-0.5*affin_x(0.7745967,X[0],h1))*(-3*affin_y(0,Y[0],h2)-affin_y(0,Y[0],h2)*affin_y(0,Y[0],h2))
                   *np.exp(affin_x(0.7745967,X[0],h1)+affin_y(0,Y[0],h2)))*pus0(0.7745967,0)+\
          (25/81)*(-affin_y(0.7745967,Y[0],h2)*(1-affin_y(0.7745967,Y[0],h2))*
                   (1-affin_x(0.7745967,X[0],h1)-0.5*affin_x(0.7745967,X[0],h1)*affin_x(0.7745967,X[0],h1))*
                   np.exp(affin_x(0.7745967,X[0],h1)+affin_y(0.7745967,Y[0],h2))-affin_x(0.7745967,X[0],h1)*
                   (1-0.5*affin_x(0.7745967,X[0],h1))*(-3*affin_y(0.7745967,Y[0],h2)-affin_y(0.7745967,Y[0],h2)*affin_y(0.7745967,Y[0],h2))
                   *np.exp(affin_x(0.7745967,X[0],h1)+affin_y(0.7745967,Y[0],h2)))*pus0(0.7745967,0.7745967)
    elif beta==1:
        r=(25/81)*(-affin_y(-0.7745967,Y[0],h2)*(1-affin_y(-0.7745967,Y[0],h2))*
                   (1-affin_x(-0.7745967,X[0],h1)-0.5*affin_x(-0.7745967,X[0],h1)*affin_x(-0.7745967,X[0],h1))*
                   np.exp(affin_x(-0.7745967,X[0],h1)+affin_y(-0.7745967,Y[0],h2))-affin_x(-0.7745967,X[0],h1)*
                   (1-0.5*affin_x(-0.7745967,X[0],h1))*(-3*affin_y(-0.7745967,Y[0],h2)-affin_y(-0.7745967,Y[0],h2)*affin_y(-0.7745967,Y[0],h2))
                   *np.exp(affin_x(-0.7745967,X[0],h1)+affin_y(-0.7745967,Y[0],h2)))*pus1(-0.7745967,-0.7745967)+\
          (40/81)*(-affin_y(0,Y[0],h2)*(1-affin_y(-0,Y[0],h2))*
                   (1-affin_x(-0.7745967,X[0],h1)-0.5*affin_x(-0.7745967,X[0],h1)*affin_x(-0.7745967,X[0],h1))*
                   np.exp(affin_x(-0.7745967,X[0],h1)+affin_y(0,Y[0],h2))-affin_x(-0.7745967,X[0],h1)*
                   (1-0.5*affin_x(-0.7745967,X[0],h1))*(-3*affin_y(0,Y[0],h2)-affin_y(0,Y[0],h2)*affin_y(0,Y[0],h2))
                   *np.exp(affin_x(-0.7745967,X[0],h1)+affin_y(0,Y[0],h2)))*pus1(-0.7745967,0)+\
          (25/81)*(-affin_y(0.7745967,Y[0],h2)*(1-affin_y(0.7745967,Y[0],h2))*
                   (1-affin_x(-0.7745967,X[0],h1)-0.5*affin_x(-0.7745967,X[0],h1)*affin_x(-0.7745967,X[0],h1))*
                   np.exp(affin_x(-0.7745967,X[0],h1)+affin_y(0.7745967,Y[0],h2))-affin_x(-0.7745967,X[0],h1)*
                   (1-0.5*affin_x(-0.7745967,X[0],h1))*(-3*affin_y(0.7745967,Y[0],h2)-affin_y(0.7745967,Y[0],h2)*affin_y(0.7745967,Y[0],h2))
                   *np.exp(affin_x(-0.7745967,X[0],h1)+affin_y(0.7745967,Y[0],h2)))*pus1(-0.7745967,0.7745967)+\
          (40/81)*(-affin_y(-0.7745967,Y[0],h2)*(1-affin_y(-0.7745967,Y[0],h2))*
                   (1-affin_x(0,X[0],h1)-0.5*affin_x(0,X[0],h1)*affin_x(0,X[0],h1))*
                   np.exp(affin_x(0,X[0],h1)+affin_y(-0.7745967,Y[0],h2))-affin_x(0,X[0],h1)*
                   (1-0.5*affin_x(0,X[0],h1))*(-3*affin_y(-0.7745967,Y[0],h2)-affin_y(-0.7745967,Y[0],h2)*affin_y(-0.7745967,Y[0],h2))
                   *np.exp(affin_x(0,X[0],h1)+affin_y(-0.7745967,Y[0],h2)))*pus1(0,-0.7745967)+\
          (64/81)*(-affin_y(0,Y[0],h2)*(1-affin_y(0,Y[0],h2))*
                   (1-affin_x(0,X[0],h1)-0.5*affin_x(0,X[0],h1)*affin_x(0,X[0],h1))*
                   np.exp(affin_x(0,X[0],h1)+affin_y(0,Y[0],h2))-affin_x(0,X[0],h1)*
                   (1-0.5*affin_x(0,X[0],h1))*(-3*affin_y(0,Y[0],h2)-affin_y(0,Y[0],h2)*affin_y(0,Y[0],h2))
                   *np.exp(affin_x(0,X[0],h1)+affin_y(0,Y[0],h2)))*pus1(0,0)+\
          (40/81)*(-affin_y(0.7745967,Y[0],h2)*(1-affin_y(0.7745967,Y[0],h2))*
                   (1-affin_x(0,X[0],h1)-0.5*affin_x(0,X[0],h1)*affin_x(0,X[0],h1))*
                   np.exp(affin_x(0,X[0],h1)+affin_y(0.7745967,Y[0],h2))-affin_x(0,X[0],h1)*
                   (1-0.5*affin_x(0,X[0],h1))*(-3*affin_y(0.7745967,Y[0],h2)-affin_y(0.7745967,Y[0],h2)*affin_y(0.7745967,Y[0],h2))
                   *np.exp(affin_x(0,X[0],h1)+affin_y(0.7745967,Y[0],h2)))*pus1(0,0.7745967)+\
          (25/81)*(-affin_y(-0.7745967,Y[0],h2)*(1-affin_y(-0.7745967,Y[0],h2))*
                   (1-affin_x(0.7745967,X[0],h1)-0.5*affin_x(0.7745967,X[0],h1)*affin_x(0.7745967,X[0],h1))*
                   np.exp(affin_x(0.7745967,X[0],h1)+affin_y(-0.7745967,Y[0],h2))-affin_x(0.7745967,X[0],h1)*
                   (1-0.5*affin_x(0.7745967,X[0],h1))*(-3*affin_y(-0.7745967,Y[0],h2)-affin_y(-0.7745967,Y[0],h2)*affin_y(-0.7745967,Y[0],h2))
                   *np.exp(affin_x(0.7745967,X[0],h1)+affin_y(-0.7745967,Y[0],h2)))*pus1(0.7745967,-0.7745967)+\
          (40/81)*(-affin_y(0,Y[0],h2)*(1-affin_y(-0,Y[0],h2))*
                   (1-affin_x(0.7745967,X[0],h1)-0.5*affin_x(0.7745967,X[0],h1)*affin_x(0.7745967,X[0],h1))*
                   np.exp(affin_x(0.7745967,X[0],h1)+affin_y(0,Y[0],h2))-affin_x(0.7745967,X[0],h1)*
                   (1-0.5*affin_x(0.7745967,X[0],h1))*(-3*affin_y(0,Y[0],h2)-affin_y(0,Y[0],h2)*affin_y(0,Y[0],h2))
                   *np.exp(affin_x(0.7745967,X[0],h1)+affin_y(0,Y[0],h2)))*pus1(0.7745967,0)+\
          (25/81)*(-affin_y(0.7745967,Y[0],h2)*(1-affin_y(0.7745967,Y[0],h2))*
                   (1-affin_x(0.7745967,X[0],h1)-0.5*affin_x(0.7745967,X[0],h1)*affin_x(0.7745967,X[0],h1))*
                   np.exp(affin_x(0.7745967,X[0],h1)+affin_y(0.7745967,Y[0],h2))-affin_x(0.7745967,X[0],h1)*
                   (1-0.5*affin_x(0.7745967,X[0],h1))*(-3*affin_y(0.7745967,Y[0],h2)-affin_y(0.7745967,Y[0],h2)*affin_y(0.7745967,Y[0],h2))
                   *np.exp(affin_x(0.7745967,X[0],h1)+affin_y(0.7745967,Y[0],h2)))*pus1(0.7745967,0.7745967)
    elif beta==2:
        r=(25/81)*(-affin_y(-0.7745967,Y[0],h2)*(1-affin_y(-0.7745967,Y[0],h2))*
                   (1-affin_x(-0.7745967,X[0],h1)-0.5*affin_x(-0.7745967,X[0],h1)*affin_x(-0.7745967,X[0],h1))*
                   np.exp(affin_x(-0.7745967,X[0],h1)+affin_y(-0.7745967,Y[0],h2))-affin_x(-0.7745967,X[0],h1)*
                   (1-0.5*affin_x(-0.7745967,X[0],h1))*(-3*affin_y(-0.7745967,Y[0],h2)-affin_y(-0.7745967,Y[0],h2)*affin_y(-0.7745967,Y[0],h2))
                   *np.exp(affin_x(-0.7745967,X[0],h1)+affin_y(-0.7745967,Y[0],h2)))*pus2(-0.7745967,-0.7745967)+\
          (40/81)*(-affin_y(0,Y[0],h2)*(1-affin_y(-0,Y[0],h2))*
                   (1-affin_x(-0.7745967,X[0],h1)-0.5*affin_x(-0.7745967,X[0],h1)*affin_x(-0.7745967,X[0],h1))*
                   np.exp(affin_x(-0.7745967,X[0],h1)+affin_y(0,Y[0],h2))-affin_x(-0.7745967,X[0],h1)*
                   (1-0.5*affin_x(-0.7745967,X[0],h1))*(-3*affin_y(0,Y[0],h2)-affin_y(0,Y[0],h2)*affin_y(0,Y[0],h2))
                   *np.exp(affin_x(-0.7745967,X[0],h1)+affin_y(0,Y[0],h2)))*pus2(-0.7745967,0)+\
          (25/81)*(-affin_y(0.7745967,Y[0],h2)*(1-affin_y(0.7745967,Y[0],h2))*
                   (1-affin_x(-0.7745967,X[0],h1)-0.5*affin_x(-0.7745967,X[0],h1)*affin_x(-0.7745967,X[0],h1))*
                   np.exp(affin_x(-0.7745967,X[0],h1)+affin_y(0.7745967,Y[0],h2))-affin_x(-0.7745967,X[0],h1)*
                   (1-0.5*affin_x(-0.7745967,X[0],h1))*(-3*affin_y(0.7745967,Y[0],h2)-affin_y(0.7745967,Y[0],h2)*affin_y(0.7745967,Y[0],h2))
                   *np.exp(affin_x(-0.7745967,X[0],h1)+affin_y(0.7745967,Y[0],h2)))*pus2(-0.7745967,0.7745967)+\
          (40/81)*(-affin_y(-0.7745967,Y[0],h2)*(1-affin_y(-0.7745967,Y[0],h2))*
                   (1-affin_x(0,X[0],h1)-0.5*affin_x(0,X[0],h1)*affin_x(0,X[0],h1))*
                   np.exp(affin_x(0,X[0],h1)+affin_y(-0.7745967,Y[0],h2))-affin_x(0,X[0],h1)*
                   (1-0.5*affin_x(0,X[0],h1))*(-3*affin_y(-0.7745967,Y[0],h2)-affin_y(-0.7745967,Y[0],h2)*affin_y(-0.7745967,Y[0],h2))
                   *np.exp(affin_x(0,X[0],h1)+affin_y(-0.7745967,Y[0],h2)))*pus2(0,-0.7745967)+\
          (64/81)*(-affin_y(0,Y[0],h2)*(1-affin_y(0,Y[0],h2))*
                   (1-affin_x(0,X[0],h1)-0.5*affin_x(0,X[0],h1)*affin_x(0,X[0],h1))*
                   np.exp(affin_x(0,X[0],h1)+affin_y(0,Y[0],h2))-affin_x(0,X[0],h1)*
                   (1-0.5*affin_x(0,X[0],h1))*(-3*affin_y(0,Y[0],h2)-affin_y(0,Y[0],h2)*affin_y(0,Y[0],h2))
                   *np.exp(affin_x(0,X[0],h1)+affin_y(0,Y[0],h2)))*pus2(0,0)+\
          (40/81)*(-affin_y(0.7745967,Y[0],h2)*(1-affin_y(0.7745967,Y[0],h2))*
                   (1-affin_x(0,X[0],h1)-0.5*affin_x(0,X[0],h1)*affin_x(0,X[0],h1))*
                   np.exp(affin_x(0,X[0],h1)+affin_y(0.7745967,Y[0],h2))-affin_x(0,X[0],h1)*
                   (1-0.5*affin_x(0,X[0],h1))*(-3*affin_y(0.7745967,Y[0],h2)-affin_y(0.7745967,Y[0],h2)*affin_y(0.7745967,Y[0],h2))
                   *np.exp(affin_x(0,X[0],h1)+affin_y(0.7745967,Y[0],h2)))*pus2(0,0.7745967)+\
          (25/81)*(-affin_y(-0.7745967,Y[0],h2)*(1-affin_y(-0.7745967,Y[0],h2))*
                   (1-affin_x(0.7745967,X[0],h1)-0.5*affin_x(0.7745967,X[0],h1)*affin_x(0.7745967,X[0],h1))*
                   np.exp(affin_x(0.7745967,X[0],h1)+affin_y(-0.7745967,Y[0],h2))-affin_x(0.7745967,X[0],h1)*
                   (1-0.5*affin_x(0.7745967,X[0],h1))*(-3*affin_y(-0.7745967,Y[0],h2)-affin_y(-0.7745967,Y[0],h2)*affin_y(-0.7745967,Y[0],h2))
                   *np.exp(affin_x(0.7745967,X[0],h1)+affin_y(-0.7745967,Y[0],h2)))*pus2(0.7745967,-0.7745967)+\
          (40/81)*(-affin_y(0,Y[0],h2)*(1-affin_y(-0,Y[0],h2))*
                   (1-affin_x(0.7745967,X[0],h1)-0.5*affin_x(0.7745967,X[0],h1)*affin_x(0.7745967,X[0],h1))*
                   np.exp(affin_x(0.7745967,X[0],h1)+affin_y(0,Y[0],h2))-affin_x(0.7745967,X[0],h1)*
                   (1-0.5*affin_x(0.7745967,X[0],h1))*(-3*affin_y(0,Y[0],h2)-affin_y(0,Y[0],h2)*affin_y(0,Y[0],h2))
                   *np.exp(affin_x(0.7745967,X[0],h1)+affin_y(0,Y[0],h2)))*pus2(0.7745967,0)+\
          (25/81)*(-affin_y(0.7745967,Y[0],h2)*(1-affin_y(0.7745967,Y[0],h2))*
                   (1-affin_x(0.7745967,X[0],h1)-0.5*affin_x(0.7745967,X[0],h1)*affin_x(0.7745967,X[0],h1))*
                   np.exp(affin_x(0.7745967,X[0],h1)+affin_y(0.7745967,Y[0],h2))-affin_x(0.7745967,X[0],h1)*
                   (1-0.5*affin_x(0.7745967,X[0],h1))*(-3*affin_y(0.7745967,Y[0],h2)-affin_y(0.7745967,Y[0],h2)*affin_y(0.7745967,Y[0],h2))
                   *np.exp(affin_x(0.7745967,X[0],h1)+affin_y(0.7745967,Y[0],h2)))*pus2(0.7745967,0.7745967)
    elif beta==3:
        r=(25/81)*(-affin_y(-0.7745967,Y[0],h2)*(1-affin_y(-0.7745967,Y[0],h2))*
                   (1-affin_x(-0.7745967,X[0],h1)-0.5*affin_x(-0.7745967,X[0],h1)*affin_x(-0.7745967,X[0],h1))*
                   np.exp(affin_x(-0.7745967,X[0],h1)+affin_y(-0.7745967,Y[0],h2))-affin_x(-0.7745967,X[0],h1)*
                   (1-0.5*affin_x(-0.7745967,X[0],h1))*(-3*affin_y(-0.7745967,Y[0],h2)-affin_y(-0.7745967,Y[0],h2)*affin_y(-0.7745967,Y[0],h2))
                   *np.exp(affin_x(-0.7745967,X[0],h1)+affin_y(-0.7745967,Y[0],h2)))*pus3(-0.7745967,-0.7745967)+\
          (40/81)*(-affin_y(0,Y[0],h2)*(1-affin_y(-0,Y[0],h2))*
                   (1-affin_x(-0.7745967,X[0],h1)-0.5*affin_x(-0.7745967,X[0],h1)*affin_x(-0.7745967,X[0],h1))*
                   np.exp(affin_x(-0.7745967,X[0],h1)+affin_y(0,Y[0],h2))-affin_x(-0.7745967,X[0],h1)*
                   (1-0.5*affin_x(-0.7745967,X[0],h1))*(-3*affin_y(0,Y[0],h2)-affin_y(0,Y[0],h2)*affin_y(0,Y[0],h2))
                   *np.exp(affin_x(-0.7745967,X[0],h1)+affin_y(0,Y[0],h2)))*pus3(-0.7745967,0)+\
          (25/81)*(-affin_y(0.7745967,Y[0],h2)*(1-affin_y(0.7745967,Y[0],h2))*
                   (1-affin_x(-0.7745967,X[0],h1)-0.5*affin_x(-0.7745967,X[0],h1)*affin_x(-0.7745967,X[0],h1))*
                   np.exp(affin_x(-0.7745967,X[0],h1)+affin_y(0.7745967,Y[0],h2))-affin_x(-0.7745967,X[0],h1)*
                   (1-0.5*affin_x(-0.7745967,X[0],h1))*(-3*affin_y(0.7745967,Y[0],h2)-affin_y(0.7745967,Y[0],h2)*affin_y(0.7745967,Y[0],h2))
                   *np.exp(affin_x(-0.7745967,X[0],h1)+affin_y(0.7745967,Y[0],h2)))*pus3(-0.7745967,0.7745967)+\
          (40/81)*(-affin_y(-0.7745967,Y[0],h2)*(1-affin_y(-0.7745967,Y[0],h2))*
                   (1-affin_x(0,X[0],h1)-0.5*affin_x(0,X[0],h1)*affin_x(0,X[0],h1))*
                   np.exp(affin_x(0,X[0],h1)+affin_y(-0.7745967,Y[0],h2))-affin_x(0,X[0],h1)*
                   (1-0.5*affin_x(0,X[0],h1))*(-3*affin_y(-0.7745967,Y[0],h2)-affin_y(-0.7745967,Y[0],h2)*affin_y(-0.7745967,Y[0],h2))
                   *np.exp(affin_x(0,X[0],h1)+affin_y(-0.7745967,Y[0],h2)))*pus3(0,-0.7745967)+\
          (64/81)*(-affin_y(0,Y[0],h2)*(1-affin_y(0,Y[0],h2))*
                   (1-affin_x(0,X[0],h1)-0.5*affin_x(0,X[0],h1)*affin_x(0,X[0],h1))*
                   np.exp(affin_x(0,X[0],h1)+affin_y(0,Y[0],h2))-affin_x(0,X[0],h1)*
                   (1-0.5*affin_x(0,X[0],h1))*(-3*affin_y(0,Y[0],h2)-affin_y(0,Y[0],h2)*affin_y(0,Y[0],h2))
                   *np.exp(affin_x(0,X[0],h1)+affin_y(0,Y[0],h2)))*pus3(0,0)+\
          (40/81)*(-affin_y(0.7745967,Y[0],h2)*(1-affin_y(0.7745967,Y[0],h2))*
                   (1-affin_x(0,X[0],h1)-0.5*affin_x(0,X[0],h1)*affin_x(0,X[0],h1))*
                   np.exp(affin_x(0,X[0],h1)+affin_y(0.7745967,Y[0],h2))-affin_x(0,X[0],h1)*
                   (1-0.5*affin_x(0,X[0],h1))*(-3*affin_y(0.7745967,Y[0],h2)-affin_y(0.7745967,Y[0],h2)*affin_y(0.7745967,Y[0],h2))
                   *np.exp(affin_x(0,X[0],h1)+affin_y(0.7745967,Y[0],h2)))*pus3(0,0.7745967)+\
          (25/81)*(-affin_y(-0.7745967,Y[0],h2)*(1-affin_y(-0.7745967,Y[0],h2))*
                   (1-affin_x(0.7745967,X[0],h1)-0.5*affin_x(0.7745967,X[0],h1)*affin_x(0.7745967,X[0],h1))*
                   np.exp(affin_x(0.7745967,X[0],h1)+affin_y(-0.7745967,Y[0],h2))-affin_x(0.7745967,X[0],h1)*
                   (1-0.5*affin_x(0.7745967,X[0],h1))*(-3*affin_y(-0.7745967,Y[0],h2)-affin_y(-0.7745967,Y[0],h2)*affin_y(-0.7745967,Y[0],h2))
                   *np.exp(affin_x(0.7745967,X[0],h1)+affin_y(-0.7745967,Y[0],h2)))*pus3(0.7745967,-0.7745967)+\
          (40/81)*(-affin_y(0,Y[0],h2)*(1-affin_y(-0,Y[0],h2))*
                   (1-affin_x(0.7745967,X[0],h1)-0.5*affin_x(0.7745967,X[0],h1)*affin_x(0.7745967,X[0],h1))*
                   np.exp(affin_x(0.7745967,X[0],h1)+affin_y(0,Y[0],h2))-affin_x(0.7745967,X[0],h1)*
                   (1-0.5*affin_x(0.7745967,X[0],h1))*(-3*affin_y(0,Y[0],h2)-affin_y(0,Y[0],h2)*affin_y(0,Y[0],h2))
                   *np.exp(affin_x(0.7745967,X[0],h1)+affin_y(0,Y[0],h2)))*pus3(0.7745967,0)+\
          (25/81)*(-affin_y(0.7745967,Y[0],h2)*(1-affin_y(0.7745967,Y[0],h2))*
                   (1-affin_x(0.7745967,X[0],h1)-0.5*affin_x(0.7745967,X[0],h1)*affin_x(0.7745967,X[0],h1))*
                   np.exp(affin_x(0.7745967,X[0],h1)+affin_y(0.7745967,Y[0],h2))-affin_x(0.7745967,X[0],h1)*
                   (1-0.5*affin_x(0.7745967,X[0],h1))*(-3*affin_y(0.7745967,Y[0],h2)-affin_y(0.7745967,Y[0],h2)*affin_y(0.7745967,Y[0],h2))
                   *np.exp(affin_x(0.7745967,X[0],h1)+affin_y(0.7745967,Y[0],h2)))*pus3(0.7745967,0.7745967)
    return r

def affin_x(x,x0,h1):
    # if h1==0:
    #     print("h1==0,warning! ")
    return 0.5*h1*(1+x)+x0
def affin_y(y,y0,h2):
    # if h2==0:
    #     print("h2==0, warning! ")
    return 0.5*h2*(1+y)+y0

def pus0(x,y):
    return 0.25*(1-x-y+x*y)
def pus1(x,y):
    return 0.25*(1+x-y-x*y)
def pus2(x,y):
    return 0.25*(1+x+y+x*y)
def pus3(x,y):
    return 0.25*(1-x+y-x*y)

def U_analy(x,y):
    return x*y*(1-0.5*x)*(1-y)*np.exp(x+y)

def Wn(N1,N2,u,n_e):
    """
    Wn is the approximation function of U_analy
    :param u: numerical solution computed by A and b
    :param beta: order of basis function
    :param n_e: order of finite elements
    :return: approximation value of U_analy
    """
    Nlb=4
    x=np.array([0]*Nlb,dtype=float)
    y=np.array([0]*Nlb,dtype=float)
    #x=np.array([-0.7745967,-0.7745967,-0.7745967,0,0,0,0.7745967,0.7745967,0.7745967])
    #y=np.array([-0.7745967,0,0.7745967]*3)
    ub=[0]*Nlb
    index_lis=T_function(N1,N2,n_e)
    for i in range(Nlb):
        x[i],y[i]=P_function(N1,N2,index_lis[i])
        ub[i]=u[int(index_lis[i])]
    h1=x[1]-x[0]
    h2=y[3]-y[0]
    #print("h1 is: ",h1,"h2 is: ",h2)
    xm=0.5*(x[0]+x[1])
    ym=0.5*(y[0]+y[3])
    x_t=(1/h1)*(2*xm-2*x[0]-h1)
    y_t=(1/h2)*(2*ym-2*y[0]-h2)
    W=ub[0]*pus0(x_t,y_t)+ub[1]*pus1(x_t,y_t)+ub[2]*pus2(x_t,y_t)+ub[3]*pus3(x_t,y_t)
    return W

def check_result(N1,N2,u,x,y):
    X = np.linspace(-1, 1, N1+1)
    Y = np.linspace(-1, 1, N2+1)
    x_ind=np.argwhere(X==x)
    y_ind=np.argwhere(Y==y)
    print("x_ind: ",x_ind,"y_ind: ",y_ind)
    mat=u.reshape((N1+1,N2+1))
    mat1=mat.T
    return mat1[x_ind,y_ind]


def U_loc(N1,N2,n_e):
    # x=np.array([-0.7745967,-0.7745967,-0.7745967,0,0,0,0.7745967,0.7745967,0.7745967])
    # y=np.array([-0.7745967,0,0.7745967]*3)
    x=np.array([0]*4,dtype=float)
    y=np.array([0]*4,dtype=float)
    index_lis=T_function(N1,N2,n_e)
    for i in range(4):
        x[i],y[i]=P_function(N1,N2,index_lis[i])
    x0=0.5*(x[0]+x[1])
    y0=0.5*(y[0]+y[3])
    result=U_analy(x0,y0)
    return result

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

i=32
N1=i
N2=i
A=get_stiffness_mat(N1,N2)
# print("Det(A)= ",np.linalg.det(A))
# print("A is:",'\n',A)
b=get_load_vector(N1,N2)
# print("The b is: ",'\n',b)
inv_A=np.linalg.inv(A)
u=np.dot(inv_A,b)
print("u is: ",'\n',u)
err=get_err(N1,N2,u)
print("The err is: ",err)
