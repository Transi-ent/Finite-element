import numpy as np
from scipy import integrate as inte


def get_stiffness_mat(N1,N2):
    """
    :param Nlb: You can set the value of Nlb(number of local basis function), Nlb=3 here
    :param Nb: number of unknowns, or of finite element nodes
    :param N: number of finite elements
    :return: stiffness matrix
    """
    Nb=(N1+1)*(N2+1)
    nbn=2*(N1+N2)
    N=2*N1*N2                              # TODO: Number of finite elemnet is 2*N1*N2
    A=np.zeros((Nb,Nb))
    for n in range(N):
        for alpha in range(3):
            for beta in range(3):
                r=gauss_quadrature_stiff(N1,N2,n,alpha,beta)
                A[Tb_function(N1,N2,beta,n)-1,Tb_function(N1,N2,alpha,n)-1]+=r
    # Deal with the boundary nodes
    for k in range(nbn):
        i=boundaryedges(N1,N2,k)-1
        A[i,:]=0
        A[i,i]=1
    return A


def gauss_quadrature_stiff(N1,N2,n_e,alpha,beta):
    x=[0,0,0]
    y=[0,0,0]
    for i in range(3):
        glo_index_n=Tb_function(N1,N2,i,n_e)-1
        x[i],y[i]=Pb_function(N1,N2,glo_index_n)

    J=(x[1]-x[0])*(y[2]-y[0])-(x[2]-x[0])*(y[1]-y[0])
    area=0.5*(x[1]-x[0])*(y[1]-y[0])
    if alpha==0 and beta==0:
        coeff=((y[2]-y[1])/J)**2+((x[1]-x[2])/J)**2
    elif (alpha==0 and beta==1) or (alpha==1 and beta==0):
        coeff=-(y[2]-y[1])*(y[2]-y[0])/(J**2)-(x[1]-x[2])*(x[0]-x[2])/(J**2)
    elif (alpha==0 and beta==2) or (alpha==2 and beta==0):
        coeff=-(y[2]-y[1])*(y[0]-y[1])/(J**2)-(x[1]-x[2])*(x[1]-x[0])/(J**2)
    elif (alpha==1 and beta==2) or (alpha==2 and beta==1):
        coeff=((y[2]-y[0])*(y[0]-y[1])+(x[0]-x[2])*(x[1]-x[0]))/(J**2)
    elif alpha==1 and beta==1:
        coeff=((y[2]-y[0])**2+(x[0]-x[2])**2)/(J**2)
    elif alpha==2 and beta==2:
        coeff=((y[0]-y[1])**2+(x[1]-x[0])**2)/(J**2)
    else:
        print("Wrong input(alpha or beta)")
    r=area*coeff
    return r


def Pb_function(N1,N2,n):
    """
    SET left=-1, right=1, top=1, bottom=-1
    :param Nb:
    :param n:
    :return: Coordinates of n-th global finite element node
    """
    left=-1
    right=1
    bottom=-1
    top=1
    Nb=(N1+1)*(N2+1)
    Pb=np.zeros((2,Nb))
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
    T=np.ones((Nlb,N))
    T[:,0]=[1,N2+2,2]
    T[:,1]=[2,2+N2,3+N2]
    for i in range(2,2*N2,2):
        T[:,i]=T[:,i-2]+[1,1,1]

    for j in range(3,2*N2,2):
        T[:,j]=T[:,j-2]+[1,1,1]

    for k in range(N2*2,N,2*N2):
        T[:,k:k+2*N2]=T[:,k-2*N2:k]+np.ones((3,2*N2))*(N2+1)

    #print("The Tb is: ",'\n',T)
    index=T[alpha,n]
    return int(index)


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


def dealWith_Boundary():
    return


def get_load_vector(N1,N2):
    """
    :param Nlb: You can set the value of Nlb(number of local basis function), Nlb=3 here
    :param Nb: number of finite element node(unknowns)
    :param N: number of finite of element
    :return: load vector
    """
    nbn=2*(N1+N2)
    N=2*N1*N2                                            #TODO: Number of finite element is 2*N1*N2(Triangle element)
    Nb=(N1+1)*(N2+1)
    b=np.zeros((Nb,1))
    for n in range(N):
        for beta in range(3):
            r=gauss_quadrature_load(N1,N2,n,beta)
            b[Tb_function(N1,N2,beta,n)-1,0]+=r
    # Deal with the boundary nodes
    for k in range(nbn):
        i=boundaryedges(N1,N2,k)-1
        x,y=Pb_function(N1,N2,i)
        b[i,0]=Boundary_fun(x,y)
    return b


def Boundary_fun(x,y):
    if x==-1:
        return -1.5*y*(1-y)*np.exp(-1+y)
    elif x==1:
        return 0.5*y*(1-y)*np.exp(1+y)
    elif y==-1:
        return -2*x*(1-0.5*x)*np.exp(x-1)
    elif y==1:
        return 0


def gauss_quadrature_load(N1,N2,n,beta):
    Nlb=3
    X=[0]*Nlb
    Y=[0]*Nlb
    for i in range(3):
        index=Tb_function(N1,N2,i,n)-1
        X[i],Y[i]=Pb_function(N1,N2,index)
    J=(X[1]-X[0])*(Y[2]-Y[0])-(X[2]-X[0])*(Y[1]-Y[0])
    if J==0:
        print("Warning!!! J==0, Please check. ")
        return None
    if beta==0:
        res_integration,error=inte.dblquad(lambda x,y:(-y*(1-y)*(1-x-0.5*x**2)*np.exp(x+y)-x*(1-0.5*x)*(-3*y-y**2)*np.exp(x+y))
                                                      *((((y-Y[0])*(X[2]-X[1])+(x-X[0])*(Y[1]-Y[2]))/J)+1),
                                           X[0],X[1],
                                           lambda x:((X[1]-x)+Y[1] if(Y[0]==Y[1]) else Y[0]),
                                           lambda x:((X[1]-x)+Y[1] if (Y[0]==Y[2]) else Y[0]))
        return res_integration
    elif beta==1:
        res_integration,error=inte.dblquad(lambda x,y:(-y*(1-y)*(1-x-0.5*x**2)*np.exp(x+y)-x*(1-0.5*x)*(-3*y-y**2)*np.exp(x+y))
                                                      *((Y[2]-Y[0])*(x-X[0])-(X[2]-X[0])*(y-Y[0]))/J,
                                           X[0],X[1],
                                           lambda x:((X[1]-x)+Y[1] if(Y[0]==Y[1]) else Y[0]),
                                           lambda x:((X[1]-x)+Y[1] if (Y[0]==Y[2]) else Y[0]))
        return res_integration
    elif beta==2:
        res_integration,error=inte.dblquad(lambda x,y:(-y*(1-y)*(1-x-0.5*x**2)*np.exp(x+y)-x*(1-0.5*x)*(-3*y-y**2)*np.exp(x+y))
                                                      *((X[1]-X[0])*(y-Y[0])-(Y[1]-Y[0])*(x-X[0]))/J,
                                           X[0],X[1],
                                           lambda x:((X[1]-x)+Y[1] if(Y[0]==Y[1]) else Y[0]),
                                           lambda x:((X[1]-x)+Y[1] if (Y[0]==Y[2]) else Y[0]))
        return res_integration


def f(x,y):
    """ f is the trail function! """
    return (-y*(1-y)*(1-x-0.5*x**2)*np.exp(x+y)-x*(1-0.5*x)*(-3*y-y**2)*np.exp(x+y))


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


def integrand(X,Y,beta,x,y):
    """ Integrand of r when taking integration of the element of load vector! """
    return f(x,y)*basis_function_load(X,Y,beta,x,y)


def upper_border(X,Y,x):
    if Y[0]==Y[1]:
        return (X[1]-x)+Y[1]
    elif Y[0]==Y[2]:
        return Y[0]


def lower_border(X,Y,x):
    if Y[0]==Y[2]:
        return (X[1]-x)+Y[1]
    elif Y[0]==Y[1]:
        return Y[0]


#print(get_load_vector(2,2))
# A=get_stiffness_mat(3,3)
# print(np.linalg.det(A))
#boundaryedges(2,2,5)
