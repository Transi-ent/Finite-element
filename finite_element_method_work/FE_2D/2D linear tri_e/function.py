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
    X=[0,0,0]
    Y=[0,0,0]
    for i in range(3):
        glo_index_n=Tb_function(N1,N2,i,n_e)-1
        X[i],Y[i]=Pb_function(N1,N2,glo_index_n)
    Ar=0.5*abs(X[0]*(Y[1]-Y[2])+X[1]*(Y[2]-Y[0])+X[2]*(Y[0]-Y[1]))
    J=(X[1]-X[0])*(Y[2]-Y[0])-(X[2]-X[0])*(Y[1]-Y[0])
    r=Ar/3*((Fx(0,0.5,Y,J,alpha)*Fx(0,0.5,Y,J,beta)+Fx(0.5,0,Y,J,alpha)*Fx(0.5,0,Y,J,beta)+Fx(0.5,0.5,Y,J,alpha)*Fx(0.5,0.5,Y,J,beta))+
            (Fy(0,0.5,X,J,alpha)*Fy(0,0.5,X,J,beta)+Fy(0.5,0,X,J,alpha)*Fy(0.5,0,X,J,beta)+Fy(0.5,0.5,X,J,alpha)*Fy(0.5,0.5,X,J,beta)))
    # J=(x[1]-x[0])*(y[2]-y[0])-(x[2]-x[0])*(y[1]-y[0])
    # area=0.5*(x[1]-x[0])*(y[1]-y[0])
    # if alpha==0 and beta==0:
    #     coeff=((y[2]-y[1])/J)**2+((x[1]-x[2])/J)**2
    # elif (alpha==0 and beta==1) or (alpha==1 and beta==0):
    #     coeff=-(y[2]-y[1])*(y[2]-y[0])/(J**2)-(x[1]-x[2])*(x[0]-x[2])/(J**2)
    # elif (alpha==0 and beta==2) or (alpha==2 and beta==0):
    #     coeff=-(y[2]-y[1])*(y[0]-y[1])/(J**2)-(x[1]-x[2])*(x[1]-x[0])/(J**2)
    # elif (alpha==1 and beta==2) or (alpha==2 and beta==1):
    #     coeff=((y[2]-y[0])*(y[0]-y[1])+(x[0]-x[2])*(x[1]-x[0]))/(J**2)
    # elif alpha==1 and beta==1:
    #     coeff=((y[2]-y[0])**2+(x[0]-x[2])**2)/(J**2)
    # elif alpha==2 and beta==2:
    #     coeff=((y[0]-y[1])**2+(x[1]-x[0])**2)/(J**2)
    # else:
    #     print("Wrong input(alpha or beta)")
    # r=area*coeff
    return r
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
    T=np.ones((Nlb,N),dtype=int)
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

def gauss_quadrature_load(N1,N2,n,beta):
    Nlb=3
    X=np.array([0]*Nlb)
    Y=np.array([0]*Nlb)
    for i in range(Nlb):
        index=Tb_function(N1,N2,i,n)-1
        X[i],Y[i]=Pb_function(N1,N2,index)
    # dx10=X[1]-X[0]
    # dx20=X[2]-X[0]
    # dy10=Y[1]-Y[0]
    # dy20=Y[2]-Y[0]
    Ak=0.5*abs(X[0]*(Y[1]-Y[2])+X[1]*(Y[2]-Y[0])+X[2]*(Y[0]-Y[1]))
    r=(Ak/3)*(F(0.5,0,X,Y)*pusi(0.5,0,beta)+F(0,0.5,X,Y)*pusi(0,0.5,beta)+F(0.5,0.5,X,Y)*pusi(0.5,0.5,beta))
    # r=(Ak/3)*(pusi(0.5,0,beta)*(-afy(0.5,0,dy10,dy20,Y[0])*(1-afy(0.5,0,dy10,dy20,Y[0]))*
    #                         (1-afx(0.5,0,dx10,dx20,X[0])-0.5*afx(0.5,0,dx10,dx20,X[0])*afx(0.5,0,dx10,dx20,X[0]))*
    #                         np.exp(afx(0.5,0,dx10,dx20,X[0])+afy(0.5,0,dy10,dy20,Y[0]))-afx(0.5,0,dx10,dx20,X[0])*
    #                         (1-0.5*afx(0.5,0,dx10,dx20,X[0]))*(-3*afy(0.5,0,dy10,dy20,Y[0])-afy(0.5,0,dy10,dy20,Y[0])*
    #                                                            afy(0.5,0,dy10,dy20,Y[0]))*np.exp(afx(0.5,0,dx10,dx20,X[0])+afy(0.5,0,dy10,dy20,Y[0])))+
    #        pusi(0.5,0.5,beta)*(-afy(0.5,0.5,dy10,dy20,Y[0])*(1-afy(0.5,0.5,dy10,dy20,Y[0]))*
    #                         (1-afx(0.5,0.5,dx10,dx20,X[0])-0.5*afx(0.5,0.5,dx10,dx20,X[0])*afx(0.5,0.5,dx10,dx20,X[0]))*
    #                         np.exp(afx(0.5,0.5,dx10,dx20,X[0])+afy(0.5,0.5,dy10,dy20,Y[0]))-afx(0.5,0.5,dx10,dx20,X[0])*
    #                         (1-0.5*afx(0.5,0.5,dx10,dx20,X[0]))*(-3*afy(0.5,0.5,dy10,dy20,Y[0])-afy(0.5,0.5,dy10,dy20,Y[0])*
    #                                                            afy(0.5,0.5,dy10,dy20,Y[0]))*np.exp(afx(0.5,0.5,dx10,dx20,X[0])+afy(0.5,0.5,dy10,dy20,Y[0])))+
    #        pusi(0,0.5,beta)*(-afy(0,0.5,dy10,dy20,Y[0])*(1-afy(0,0.5,dy10,dy20,Y[0]))*
    #                         (1-afx(0,0.5,dx10,dx20,X[0])-0.5*afx(0,0.5,dx10,dx20,X[0])*afx(0,0.5,dx10,dx20,X[0]))*
    #                         np.exp(afx(0,0.5,dx10,dx20,X[0])+afy(0,0.5,dy10,dy20,Y[0]))-afx(0,0.5,dx10,dx20,X[0])*
    #                         (1-0.5*afx(0,0.5,dx10,dx20,X[0]))*(-3*afy(0,0.5,dy10,dy20,Y[0])-afy(0,0.5,dy10,dy20,Y[0])*
    #                                                            afy(0,0.5,dy10,dy20,Y[0]))*np.exp(afx(0,0.5,dx10,dx20,X[0])+afy(0,0.5,dy10,dy20,Y[0])))  )

    return r

def afx(x,y,dx10,dx20,x0):
    return dx10*x+dx20*y+x0
def afy(x,y,dy10,dy20,y0):
    return dy10*x+dy20*y+y0
def pusi(x,y,beta):
    """The reference basis function"""
    if beta==0:
        return 1-x-y
    elif beta==1:
        return x
    elif beta==2:
        return y

def Boundary_fun(x,y):
    if x==-1:
        return -1.5*y*(1-y)*np.exp(-1+y)
    elif x==1:
        return 0.5*y*(1-y)*np.exp(1+y)
    elif y==-1:
        return -2*x*(1-0.5*x)*np.exp(x-1)
    elif y==1:
        return 0

def integrate_load(N1,N2,n,beta):
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

def F(x,y,X,Y):
    """ f is the trail function! """
    x0=X[0]*pusi(x,y,0)+X[1]*pusi(x,y,1)+X[2]*pusi(x,y,2)
    y0=Y[0]*pusi(x,y,0)+Y[1]*pusi(x,y,1)+Y[2]*pusi(x,y,2)
    return (-y0*(1-y0)*(1-x0-0.5*x0**2)*np.exp(x0+y0)-x0*(1-0.5*x0)*(-3*y0-y0**2)*np.exp(x0+y0))

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

i=16
N1=i
N2=i
A=get_stiffness_mat(N1,N2)
b=get_load_vector(N1,N2)
# print("A is: ",'\n',A)
# print("Det(A) is: ",np.linalg.det(A))
u=np.dot(np.linalg.inv(A),b)
print("u is: ",'\n',u)
err=get_error(N1,N2,u)
print("The err is: ",err)

