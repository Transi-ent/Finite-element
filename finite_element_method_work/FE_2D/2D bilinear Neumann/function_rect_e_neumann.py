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
        index=boundarynodes(N1,N2,k)
        A[index[1],:]=0
        A[index[1],index[1]]=1
    return A

def boundarynodes(N1,N2,k):
    """ The first row to store the type of boundaryedge, the second row is to store the index of global node """
    right1=[]
    top1=[]
    Nb=(N1+1)*(N2+1)
    nbn=2*(N1+N2)
    boundaryNode=np.zeros((2,nbn),dtype=int)
    boundaryNode[0,:]=[-1]*nbn
    boundaryNode[0,1:N1]=[-2]*(N1-1)
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
    return boundaryNode[:,k]

def boundaryedges(N1,N2,k):
    """
    K: k-th boundary edge of all edges
    mate is a matrix obtained to get index of element
    mat_n is a matrix obtained to get index of boundarynodes
    :param N1:
    :param N2:
    :return:
    """
    right1=[]
    top1=[]
    rightn1=[]
    topn1=[]
    nbe=2*(N1+N2)
    Nm=N1*N2
    Nn=(N1+1)*(N2+1)
    edgesMat=np.zeros((4,nbe),dtype=int)
    edgesMat[0,:]=-1
    edgesMat[0,:N1]=-2
    mate=np.arange(Nm,dtype=int).reshape((N1,N2))
    mat_n=np.arange(Nn,dtype=int).reshape((N1+1,N2+1))
    top=list(mate[0,:])
    bottom=list(mate[N1-1,:])
    left=list(mate[:,0])
    right=list(mate[:,N2-1])
    for i in range(len(right)):
        right1.append(right.pop())
    for j in range(len(top)):
        top1.append(top.pop())
    topn=list(mat_n[0,:])
    bottomn=list(mat_n[N1,:])
    leftn=list(mat_n[:,0])
    rightn=list(mat_n[:,N2])
    for i in range(len(rightn)):
        rightn1.append(rightn.pop())
    for j in range(len(topn)):
        topn1.append(topn.pop())
    # del bottom[0]
    # del right1[0]
    # del top1[0]
    # del top1[-1]
    index=left+bottom+right1+top1
    edgesMat[1,:]=index
    edgesMat[2,:]=leftn[:-1]+bottomn[:-1]+rightn1[:-1]+topn1[:-1]
    edgesMat[3,:]=leftn[1:]+bottomn[1:]+rightn1[1:]+topn1[1:]
    # print("The boundaryedges mat is: ",'\n',edgesMat)
    return edgesMat[:,k]

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
    #print("The rough b is: ",'\n',b)
    # TODO: Deal with the Neumann boundary
    V=np.zeros((Nb,1),dtype=float)
    for j in range(nbn):
        info_lis=boundaryedges(N1,N2,j)      # as nbn==nbe
        if info_lis[0]==-2:
            n_e=info_lis[1]
            for beta in range(Nlb):
                r=gauss_quadrature_load_v(n_e,beta)
                index_lis=T_function(N1,N2,n_e)
                V[index_lis[beta],0]+=r
    #print("V is: ",'\n',V)
    # TODO:Deal with the Dirichlet boundary
    for k in range(nbn):            # lis is to store the info of k-th boundarynode
        lis=boundarynodes(N1,N2,k)
        if lis[0]==-1:
            x,y=P_function(N1,N2,lis[1])
            b[lis[1],0]=Boundary_fun(x,y)
        elif lis[0]==-2:
            b[lis[1],0]=V[lis[1],0]
    return b

def gauss_quadrature_load_v(n_e,beta,):
    """ Computing the quadrature(1D) caused by Neumann boundary! """
    Nlb=4
    X=np.array([0]*Nlb,dtype=float)
    Y=np.array([0]*Nlb,dtype=float)
    index_lis=T_function(N1,N2,n_e)
    for i in range(Nlb):
        X[i],Y[i]=P_function(N1,N2,index_lis[i])
    h1=X[1]-X[0]

    r = (0.5*h1)*((5/9)*np.exp(affin_x(-0.7745967,X[0],h1)-1)*pusi(-0.7745967,-1,beta)+
                  (8/9)*np.exp(affin_x(0,X[0],h1)-1)*pusi(0,-1,beta)+
                  (5/9)*np.exp(affin_x(0.7745967,X[0],h1)-1)*pusi(0.7745967,-1,beta) )
    return r

def Boundary_fun(x,y):
    """ adding boundary condition! """
    if x==-1:
        return np.exp(-1+y)
    elif x==1:
        return np.exp(1+y)
    # TODO: Adding the Neumann boundary condition!!!
    # elif y==-1:
    #     return -2*x*(1-0.5*x)*np.exp(x-1)
    elif y==1:
        return np.exp(1+x)

def gauss_quadrature_load(N1,N2,n_e,beta):
    Nlb=4
    X=np.array([0]*Nlb,dtype=float)
    Y=np.array([0]*Nlb,dtype=float)
    index_lis=T_function(N1,N2,n_e)
    for i in range(Nlb):
        X[i],Y[i]=P_function(N1,N2,index_lis[i])
    h1=X[1]-X[0]
    h2=Y[3]-Y[0]
    r =  ((25/81)*f(-0.7745967,-0.7745967,X[0],Y[0],h1,h2)*pusi(-0.7745967,-0.7745967,beta)+
          (40/81)*f(-0.7745967,0,X[0],Y[0],h1,h2)*pusi(-0.7745967,0,beta)+
          (25/81)*f(-0.7745967,0.7745967,X[0],Y[0],h1,h2)*pusi(-0.7745967,0.7745967,beta)+
          (40/81)*f(0,-0.7745967,X[0],Y[0],h1,h2)*pusi(0,-0.7745967,beta)+
          (64/81)*f(0,0,X[0],Y[0],h1,h2)*pusi(0,0,beta)+
          (40/81)*f(0,0.7745967,X[0],Y[0],h1,h2)*pusi(0,0.7745967,beta)+
          (25/81)*f(0.7745967,-0.7745967,X[0],Y[0],h1,h2)*pusi(0.7745967,-0.7745967,beta)+
          (40/81)*f(0.7745967,0,X[0],Y[0],h1,h2)*pusi(0.7745967,0,beta)+
          (25/81)*f(0.7745967,0.7745967,X[0],Y[0],h1,h2)*pusi(0.7745967,0.7745967,beta)    )*(0.25*h1*h2)
    return r

def f(x,y,x0,y0,h1,h2):
    x_lo=affin_x(x,x0,h1)
    y_lo=affin_y(y,y0,h2)
    return -2*np.exp(x_lo+y_lo)

def affin_x(x,x0,h1):
    return 0.5*h1*(1+x)+x0
def affin_y(y,y0,h2):
    # if h2==0:
    #     print("h2==0, warning! ")
    return 0.5*h2*(1+y)+y0

def pusi(x,y,beta):
    if beta==0:
        return 0.25*(1-x-y+x*y)
    elif beta==1:
        return 0.25*(1+x-y-x*y)
    elif beta==2:
        return 0.25*(1+x+y+x*y)
    elif beta==3:
        return 0.25*(1-x+y-x*y)

def U_analy(x,y):
    return np.exp(x+y)

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
    W=ub[0]*pusi(x_t,y_t,0)+ub[1]*pusi(x_t,y_t,1)+ub[2]*pusi(x_t,y_t,2)+ub[3]*pusi(x_t,y_t,3)
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

i=16
N1=i
N2=i
# boundaryedges(N1,N2,2)
A=get_stiffness_mat(N1,N2)
print("Det(A)= ",np.linalg.det(A))
# print("A is:",'\n',A)
b=get_load_vector(N1,N2)
# print("The b is: ",'\n',b)
inv_A=np.linalg.inv(A)
u=np.dot(inv_A,b)
print("u is: ",'\n',u)
err=get_err(N1,N2,u)
print("The err is: ",err)
