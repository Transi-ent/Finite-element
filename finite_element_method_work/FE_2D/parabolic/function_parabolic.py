#triangle element
import numpy as np

def get_stiffness_mat(N1,N2,bo):
    """
    if bo==0: to obtain A, elif bo==1: to obtain M
    :param Nlb: You can set the value of Nlb(number of local basis function), Nlb=3 here
    :param Nb: number of unknowns, or of finite element nodes
    :param N: number of finite elements
    :return: stiffness matrix
    """
    Nb=(N1+1)*(N2+1)
    nbn=2*(N1+N2)
    N=2*N1*N2                              # TODO: Number of finite elemnet is 2*N1*N2
    AorM=np.zeros((Nb,Nb), dtype=float)
    for n in range(N):
        for alpha in range(3):
            for beta in range(3):
                r=gauss_quadrature_stiff(N1,N2,n,alpha,beta,bo)
                AorM[Tb_function(N1,N2,beta,n)-1,Tb_function(N1,N2,alpha,n)-1]+=r
    #TODO: Deal with the boundary nodes
    for k in range(nbn):
        i=boundaryedges(N1,N2,k)-1
        AorM[i,:]=0
        AorM[i,i]=1
    return AorM

def gauss_quadrature_stiff(N1,N2,n_e,alpha,beta,bo):
    # if bo==0: to obtain A, elif bo==1: to obtain M
    X=np.array([0,0,0],dtype=float)
    Y=np.array([0,0,0],dtype=float)

    for i in range(3):
        glo_index_n=Tb_function(N1,N2,i,n_e)-1
        X[i],Y[i]=Pb_function(N1,N2,glo_index_n)
    Ar=0.5*abs(X[0]*(Y[1]-Y[2])+X[1]*(Y[2]-Y[0])+X[2]*(Y[0]-Y[1]))
    c=2
    J=(X[1]-X[0])*(Y[2]-Y[0])-(X[2]-X[0])*(Y[1]-Y[0])
    if bo==0:
        r=c*(Ar/3.0)*((Fx(0,0.5,Y,J,alpha)*Fx(0,0.5,Y,J,beta)+Fx(0.5,0,Y,J,alpha)*Fx(0.5,0,Y,J,beta)+Fx(0.5,0.5,Y,J,alpha)*Fx(0.5,0.5,Y,J,beta))+
                      (Fy(0,0.5,X,J,alpha)*Fy(0,0.5,X,J,beta)+Fy(0.5,0,X,J,alpha)*Fy(0.5,0,X,J,beta)+Fy(0.5,0.5,X,J,alpha)*Fy(0.5,0.5,X,J,beta)))
        return r
    elif bo==1:
        r=(Ar/3.0)*(phi(0,0.5,X,Y,alpha)*phi(0,0.5,X,Y,beta)+phi(0.5,0,X,Y,alpha)*phi(0.5,0,X,Y,beta)+
                    phi(0.5,0.5,X,Y,alpha)*phi(0.5,0.5,X,Y,beta))
        return r

def phi(x,y,X,Y,alpha):
    # N is a vector consisting of values of shape function at gauss points
    N=np.array([1-x-y, x, y], dtype=float)
    if alpha==0:
        return 1-np.dot(N,X)-np.dot(N,Y)
    elif alpha==1:
        return np.dot(N,X)
    elif alpha==2:
        return np.dot(N,Y)
    else:
        print("alpha must no more than 2")

def Fx(x,y,Y,J,alpha):
    """ F(x,y) is integrand, Fx(Fy) is derivative of F about x(y) """
    if alpha==0:
        return -(1/J)*(Y[2]-Y[0]+Y[0]-Y[1])
    elif alpha==1:
        return (1/J)*(Y[2]-Y[0])
    elif alpha==2:
        return (1/J)*(Y[0]-Y[1])

def Fy(x,y,X,J,alpha):
    if alpha==0:
        return (-1/J)*(X[1]-X[2])
    elif alpha==1:
        return (1/J)*(X[0]-X[2])
    elif alpha==2:
        return (1/J)*(X[1]-X[0])

def Pb_function(N1,N2,n):
    """
    SET left=0, right=2, top=0, bottom=1
    :param Nb:
    :param n:
    :return: Coordinates of n-th global finite element node
    """
    left=0
    right=2
    bottom=0
    top=1
    Nb=(N1+1)*(N2+1)
    Pb=np.zeros((2,Nb),dtype=float)
    x_array=np.linspace(left,right,N1+1)
    y_array=np.linspace(bottom,top,N2+1)
    lis=[]
    for j in x_array:
        lis+=[j]*(N2+1)
    Pb[0,:]=lis[:]
    Pb[1,:]=list(y_array)[:]*(N2+1)
    #print(" n is :",n)
    x,y=Pb[:,int(n)]
    return x,y

def Tb_function(N1,N2,alpha,n):
    """
    TODO: as the index of python start from 0, all the elements in the Tb should be subtracted 1
    for the 1D triangle element,  row of Tb==3
    :param N: number of element
    :param Nlb: number of local basis function
    :param alpha:
    :param n:
    :return:
    """
    Nlb=3
    N=2*N1*N2
    T=np.ones((Nlb,N),dtype=int)
    T[:,0]=[1,N2+2,2]
    T[:,1]=[2,2+N2,3+N2]
    for i in range(2,2*N2,2):
        T[:,i]=T[:,i-2]+[1,1,1]

    for j in range(3,2*N2,2):
        T[:,j]=T[:,j-2]+[1,1,1]

    for k in range(N2*2,N,2*N2):
        T[:,k:k+2*N2]=T[:,k-2*N2:k]+np.ones((3,2*N2))*(N2+1)
    # print("The Tb is: ",'\n',T)
    index=T[alpha,n]
    return index

def boundaryedges(N1,N2,k):
    """
    The first row stores the type of boundary,the second row store,
    the second row stores the global node index of boundary node
    :param N1:
    :param N2:
    :param k: k-th boundary
    :return: global node index of the k-th boundary node
    TODO: As index in Python start from 0, so all the global index should minus 1
    """
    lis=[]
    lis1=np.arange(2,N2+1)
    q=1
    s=1
    nbn=2*(N1+N2)
    boundary_mat=np.zeros((2,nbn))
    boundary_mat[0,:]=[-1]*nbn       # set the boundary edge type as Dirichlet by default
    for i in range(N1+1):
        boundary_mat[1,i]=(N2+1)*i+1
    global_index=(N2+1)*N1+2
    for j in range(N1+1,N1+N2+1):
        boundary_mat[1,j]=global_index
        global_index+=1
    for k1 in boundary_mat[1,:N1]:
        lis.append(k1+N2)
    for h in range(N1+N2+1,2*N1+N2+1):
        boundary_mat[1,h]=lis[-q]
        q+=1
    for g in range(2*N1+N2+1,nbn):
        boundary_mat[1,g]=lis1[-s]
        s+=1
    #print(boundary_mat)
    p=int(boundary_mat[1,int(k)])
    return p

def get_load_vector(N1,N2,t):
    """
    :param Nlb: You can set the value of Nlb(number of local basis function), Nlb=3 here
    :param Nb: number of finite element node(unknowns)
    :param N: number of finite of element
    :return: load vector
    """
    nbn=2*(N1+N2)
    N=2*N1*N2                                            #TODO: Number of finite element is 2*N1*N2(Triangle element)
    Nb=(N1+1)*(N2+1)
    b=np.zeros((Nb,1), dtype=float)
    for n in range(N):
        for beta in range(3):
            r=gauss_quadrature_load(N1,N2,n,beta,t)
            b[Tb_function(N1,N2,beta,n)-1,0]+=r
    # Deal with the boundary nodes
    for k in range(nbn):
        i=boundaryedges(N1,N2,k)-1
        x,y=Pb_function(N1,N2,i)
        b[i,0]=Boundary_fun(x,y,t)
    return b

def gauss_quadrature_load(N1,N2,n,beta,t):
    Nlb=3
    X=np.array([0]*Nlb,dtype=float)
    Y=np.array([0]*Nlb,float)
    for i in range(Nlb):
        index=Tb_function(N1,N2,i,n)-1
        X[i],Y[i]=Pb_function(N1,N2,index)

    Ak=0.5*abs(X[0]*(Y[1]-Y[2])+X[1]*(Y[2]-Y[0])+X[2]*(Y[0]-Y[1]))
    r=(Ak/3.0)*(F(0.5,0,X,Y,t)*phi(0.5,0.,X,Y,beta)+F(0,0.5,X,Y,t)*phi(0.,0.5,X,Y,beta)+F(0.5,0.5,X,Y,t)*phi(0.5,0.5,X,Y,beta))
    return r

def pusi(x,y,beta):
    """The reference basis function"""
    if beta==0:
        return 1-x-y
    elif beta==1:
        return x
    elif beta==2:
        return y

def Boundary_fun(x,y,t):
    if x==0:
        return np.exp(t+y)
    elif x==2:
        return np.exp(2+y+t)
    elif y==0:
        return np.exp(x+t)
    elif y==1:
        return np.exp(x+t+1)

def F(x,y,X,Y,t):
    """ f is the trail function! """
    x0=X[0]*pusi(x,y,0)+X[1]*pusi(x,y,1)+X[2]*pusi(x,y,2)
    y0=Y[0]*pusi(x,y,0)+Y[1]*pusi(x,y,1)+Y[2]*pusi(x,y,2)
    return -3*np.exp(x0+y0+t)

def basis_function_load(X,Y,beta,x,y):
    J=(X[1]-X[0])*(Y[2]-Y[0])-(X[2]-X[0])*(Y[1]-Y[0])
    if J==0:
        print("Warning!!! J==0, Please check. ")
        return None
    if beta==0:
        return ((((y-Y[0])*(X[2]-X[1])+(x-X[0])*(Y[1]-Y[2]))/J)+1)
    elif beta==1:
        return ((Y[2]-Y[0])*(x-X[0])-(X[2]-X[0])*(y-Y[0]))/J
    elif beta==2:
        return ((X[1]-X[0])*(y-Y[0])-(Y[1]-Y[0])*(x-X[0]))/J

def analytic_s(n):
    index_lis=[]
    X=[0,0,0]
    Y=[0,0,0]
    for i in range(3):                     # Nlb=3
        index=Tb_function(N1,N2,i,n)
        index_lis.append(index)
        X[i],Y[i]=Pb_function(N1,N2,index-1)
    x0=X[0]*pusi(0.5,0.5,0)+X[1]*pusi(0.5,0.5,1)+X[2]*pusi(0.5,0.5,2)
    y0=Y[0]*pusi(0.5,0.5,0)+Y[1]*pusi(0.5,0.5,1)+Y[2]*pusi(0.5,0.5,2)
    return x0*y0*(1-0.5*x0)*(1-y0)*np.exp(x0+y0)

def pusi_lo(x,y,X,Y,beta):
    x0=X[0]*pusi(x,y,0)+X[1]*pusi(x,y,1)+X[2]*pusi(x,y,2)
    y0=Y[0]*pusi(x,y,0)+Y[1]*pusi(x,y,1)+Y[2]*pusi(x,y,2)
    if beta==0:
        return 1-x0-y0
    elif beta==1:
        return x0
    elif beta==2:
        return y0

def wx(N1,N2,n,u):
        # J=(X[1]-X[0])*(Y[2]-Y[0])-(X[2]-X[0])*(Y[1]-Y[0])
    # Wn=u[index_lis[0]-1,0]*((((y-Y[0])*(X[2]-X[1])+(x-X[0])*(Y[1]-Y[2]))/J)+1)+\
    #    u[index_lis[1]-1,0]*((Y[2]-Y[0])*(x-X[0])-(X[2]-X[0])*(y-Y[0]))/J+\
    #    u[index_lis[2]-1,0]*((X[1]-X[0])*(y-Y[0])-(Y[1]-Y[0])*(x-X[0]))/J
    index_lis=[]
    X=[0,0,0]
    Y=[0,0,0]
    for i in range(3):                     # Nlb=3
        index=Tb_function(N1,N2,i,n)
        index_lis.append(index)
        X[i],Y[i]=Pb_function(N1,N2,index-1)
        #pos_lis.append([px,py])
    Wn=u[index_lis[0]-1,0]*pusi_lo(0.5,0.5,X,Y,0)+\
       u[index_lis[1]-1,0]*pusi_lo(0.5,0.5,X,Y,1)+\
       u[index_lis[2]-1,0]*pusi_lo(0.5,0.5,X,Y,2)
    return Wn

def get_error(N1,N2,u):
    N=2*N1*N2
    err_lis=[]
    for n in range(N):
        lis=[]
        for i in range(3):   # Nlb=3
            index=Tb_function(N1,N2,i,n)
            # x,y=Pb_function(N1,N2,index-1)
            e=abs(analytic_s(n)-wx(N1,N2,n,u))
            lis.append(e)
        err_lis.append(max(lis))
    return max(err_lis)

i=2
N1=i
N2=i
alpha=0
n=3
Tb_function(N1,N2,alpha,n)
