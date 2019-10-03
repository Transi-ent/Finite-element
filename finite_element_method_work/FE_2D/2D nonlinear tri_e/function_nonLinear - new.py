import numpy as np

def get_stiffness_mat(N1,N2):
    """
    :param Nlb: You can set the value of Nlb(number of local basis function), Nlb=6 here
    :param Nb: number of unknowns, or of finite element nodes
    :param N: number of finite elements
    :return: stiffness matrix
    """
    Nlb=6
    Nb=(2*N1+1)*(2*N2+1)
    nbn=4*(N1+N2)
    N=2*N1*N2                              # TODO: Number of finite elemnet is 2*N1*N2
    A=np.zeros((Nb,Nb),dtype=float)
    for n in range(N):
        for alpha in range(Nlb):
            for beta in range(Nlb):
                r=gauss_quadrature_stiffness(N1,N2,n,alpha,beta)
                index_lis=Tb_function(N1,N2,n)
                A[index_lis[beta],index_lis[alpha]]+=r
    # Deal with the boundary nodes
    for k in range(nbn):
        i=boundaryedges(N1,N2,k)
        A[i,:]=0
        A[i,i]=1
    return A

def gauss_quadrature_stiffness(N1,N2,n_e,alpha,beta):
    """
    :phi0: derivative of phi0(x) about x
    :param N1:
    :param N2:
    :param n_e:
    :param alpha:
    :param beta:
    :return:
    """
    Nlb=6
    X=np.array([0]*Nlb,dtype=float)
    Y=np.array([0]*Nlb,dtype=float)
    index_lis=Tb_function(N1,N2,n_e)
    for i in range(Nlb):
        X[i],Y[i]=Pb_function(N1,N2,index_lis[i])
    # print("X is: ",X)
    # print("Y is: ",Y)
    J=(X[1]-X[0])*(Y[2]-Y[0])-(X[2]-X[0])*(Y[1]-Y[0])
    if J==0:
        print("Wrong here! J==0 ")
    dy31=Y[2]-Y[0]
    dy12=Y[0]-Y[1]
    dx13=X[0]-X[2]
    dx21=X[1]-X[0]
    Ar=0.5*abs(X[0]*(Y[1]-Y[2])+X[1]*(Y[2]-Y[0])+X[2]*(Y[0]-Y[1]))
    r=(Ar/3)*(Fx(0.5,0,J,dy31,dy12,alpha)*Fx(0.5,0,J,dy31,dy12,beta)+Fx(0,0.5,J,dy31,dy12,alpha)*Fx(0,0.5,J,dy31,dy12,beta)+
              Fx(0.5,0.5,J,dy31,dy12,alpha)*Fx(0.5,0.5,J,dy31,dy12,beta)+
              Fy(0.5,0,J,dx13,dx21,alpha)*Fy(0.5,0,J,dx13,dx21,beta)+Fy(0,0.5,J,dx13,dx21,alpha)*Fy(0,0.5,J,dx13,dx21,beta)+
              Fy(0.5,0.5,J,dx13,dx21,alpha)*Fy(0.5,0.5,J,dx13,dx21,beta))

    # r=1/(6*J)*((phi(0.5,0,alpha,0)*dy31+phi(0.5,0,alpha,1)*dy12)*(phi(0.5,0,beta,0)*dy31+phi(0.5,0,beta,1)*dy12)+
    #            (phi(0.5,0.5,alpha,0)*dy31+phi(0.5,0.5,alpha,1)*dy12)*(phi(0.5,0.5,beta,0)*dy31+phi(0.5,0.5,beta,1)*dy12)+
    #            (phi(0,0.5,alpha,0)*dy31+phi(0,0.5,alpha,1)*dy12)*(phi(0,0.5,beta,0)*dy31+phi(0,0.5,beta,1)*dy12)+
    #
    #            (phi(0.5,0,alpha,0)*dx13+phi(0.5,0,alpha,1)*dx21)*(phi(0.5,0,beta,0)*dx13+phi(0.5,0,beta,1)*dx21)+
    #            (phi(0.5,0.5,alpha,0)*dx13+phi(0.5,0.5,alpha,1)*dx21)*(phi(0.5,0.5,beta,0)*dx13+phi(0.5,0.5,beta,1)*dx21)+
    #            (phi(0,0.5,alpha,0)*dx13+phi(0,0.5,alpha,1)*dx21)*(phi(0,0.5,beta,0)*dx13+phi(0,0.5,beta,1)*dx21))*\
    #   dx21*dy31
    return r
def Fx(x,y,J,dy31,dy12,beta):
    return phi(x,y,beta,0)*(dy31/J)+phi(x,y,beta,1)*(dy12/J)
def Fy(x,y,J,dx13,dx21,beta):
    return phi(x,y,beta,0)*(dx13/J)+phi(x,y,beta,1)*(dx21/J)
def phi(x,y,alpha,bo):
    """if bo==0(1), return a expression about x(y). """
    if alpha==0 :
        return 4*x+4*y-3
    elif alpha==1 and bo==0:
        return 4*x-1
    elif alpha==1 and bo==1:
        return 0
    elif alpha==2 and bo==0:
        return 0
    elif alpha==2 and bo==1:
        return 4*y-1
    elif alpha==3 and bo==0:
        return -8*x-4*y+4
    elif alpha==3 and bo==1:
        return -4*x
    elif alpha==4 and bo==0:
        return 4*y
    elif alpha==4 and bo==1:
        return 4*x
    elif alpha==5 and bo==0:
        return -4*y
    elif alpha==5 and bo==1:
        return -8*y-4*x+4

def Tb_function(N1,N2,n_e):
    N=2*N1*N2
    Nlb=6
    T=np.zeros((Nlb,N),dtype=int)
    T[:,0]=[0,2*(2*N2+1),2,  2*N2+1,2*N2+2,1]
    T[:,1]=[2,2*(2*N2+1),2*(2*N2+2),  2*N2+2,2*(2*N2+1)+1,2*N2+3]
    for k in range(2,2*N2):
        T[:,k]=T[:,k-2]+[2]*Nlb
    for i in range(1,N1):
        T[:,i*(2*N2)]=T[:,i*(2*N2)-2]+[2*N2+4]*Nlb
        T[:,i*(2*N2)+1]=T[:,i*(2*N2)-1]+[2*N2+4]*Nlb
        for j in range(1,2*N2-1):
            T[:,i*(2*N2)+1+j]=T[:,i*(2*N2)-1+j]+[2]*Nlb
    #print("The T is: ",'\n',T)
    return T[:,n_e]

def Pb_function(N1,N2,n_g):
    left=-1
    right=1
    bottom=-1
    top=1
    xlis=[]
    Nb=(2*N1+1)*(2*N2+1)
    x_array=np.linspace(left,right,2*N1+1,dtype=float)
    y_array=np.linspace(bottom,top,2*N2+1,dtype=float)
    P=np.zeros((2,Nb),dtype=float)
    for i in x_array:
        xlis+=[i]*(2*N2+1)
    P[0,:]=xlis
    P[1,:]=list(y_array)*(2*N1+1)
    #print("The P is: ",'\n',P)
    x,y=P[:,n_g]
    return x,y

def get_load_vector(N1,N2):
    """
    :param Nlb: You can set the value of Nlb(number of local basis function), Nlb=6 here
    :param Nb: number of finite element node(unknowns)
    :param N: number of finite of element
    :return: load vector
    """
    Nlb=6
    nbn=4*(N1+N2)
    N=2*N1*N2                                            #TODO: Number of finite element is 2*N1*N2(Triangle element)
    Nb=(2*N1+1)*(2*N2+1)
    b=np.zeros((Nb,1),dtype=float)
    for n in range(N):
        for beta in range(Nlb):
            r=gauss_quadrature_load(N1,N2,n,beta)
            index_lis=Tb_function(N1,N2,n)
            b[index_lis[beta],0]+=r
    # Deal with the boundary nodes
    for k in range(nbn):
        i=boundaryedges(N1,N2,k)
        x,y=Pb_function(N1,N2,i)
        b[i,0]=Boundary_fun(x,y)
    return b

def boundaryedges(N1,N2,k):
    """Return the global node index of boundary node! """
    right1=[]
    top1=[]
    Nb=(2*N1+1)*(2*N2+1)
    nbn=4*(N1+N2)
    boundaryNode=np.zeros((2,nbn),dtype=int)
    boundaryNode[0,:]=[-1]*nbn
    array=np.arange(Nb,dtype=int)
    mat=array.reshape((2*N1+1,2*N2+1))
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

def gauss_quadrature_load(N1,N2,n,beta):
    Nlb=6
    X=np.array([0]*Nlb,dtype=float)
    Y=np.array([0]*Nlb,dtype=float)
    index_lis=Tb_function(N1,N2,n)
    for i in range(Nlb):
        X[i],Y[i]=Pb_function(N1,N2,index_lis[i])
    # dx10=X[1]-X[0]
    # dx20=X[2]-X[0]
    # dy10=Y[1]-Y[0]
    # dy20=Y[2]-Y[0]
    Ar=0.5*abs(X[0]*(Y[1]-Y[2])+X[1]*(Y[2]-Y[0])+X[2]*(Y[0]-Y[1]))
    r=(Ar/3)*(F(0.5,0,X,Y)*pusi(0.5,0,beta)+F(0,0.5,X,Y)*pusi(0,0.5,beta)+F(0.5,0.5,X,Y)*pusi(0.5,0.5,beta))

    # r=1/6*(pusi(0.5,0,beta)*(-afy(0.5,0,dy10,dy20,Y[0])*(1-afy(0.5,0,dy10,dy20,Y[0]))*
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
    #                                                            afy(0,0.5,dy10,dy20,Y[0]))*np.exp(afx(0,0.5,dx10,dx20,X[0])+afy(0,0.5,dy10,dy20,Y[0]))) )*\
    #   dx10*dy20
    #TODO: 积分都不会了吗？？？
    return r
def F(x,y,X,Y):
    """ f is the trail function! """
    x0=X[0]*pusi(x,y,0)+X[1]*pusi(x,y,1)+X[2]*pusi(x,y,2)+X[3]*pusi(x,y,3)+X[4]*pusi(x,y,4)+X[5]*pusi(x,y,5)
    y0=Y[0]*pusi(x,y,0)+Y[1]*pusi(x,y,1)+Y[2]*pusi(x,y,2)+Y[3]*pusi(x,y,3)+Y[4]*pusi(x,y,4)+Y[5]*pusi(x,y,5)
    return (-y0*(1-y0)*(1-x0-0.5*x0**2)*np.exp(x0+y0)-x0*(1-0.5*x0)*(-3*y0-y0**2)*np.exp(x0+y0))
def pusi(x,y,beta):
    if beta==0:
        return 2*x**2+2*y**2+4*x*y-3*y-3*x+1
    elif beta==1:
        return 2*x**2-x
    elif beta==2:
        return 2*y**2-y
    elif beta==3:
        return -4*x**2-4*x*y+4*x
    elif beta==4:
        return 4*x*y
    elif beta==5:
        return -4*y**2-4*x*y+4*y

def Boundary_fun(x,y):
    if x==-1:
        return -1.5*y*(1-y)*np.exp(-1+y)
    elif x==1:
        return 0.5*y*(1-y)*np.exp(1+y)
    elif y==-1:
        return -2*x*(1-0.5*x)*np.exp(x-1)
    elif y==1:
        return 0

def afx(x,y,dx10,dx20,x0):
    return dx10*x+dx20*y+x0
def afy(x,y,dy10,dy20,y0):
    return dy10*x+dy20*y+y0

def get_err(N1,N2,u):
    """
    error obtained belongs to barycenter
    :param N1: number of elements in the direction of x-axis
    :param N2: number of elements in the direction of y-axis
    :param u: function value at nodes
    :return: max error in the feasible area
    """
    N=2*N1*N2
    err=[]
    for i in range(N):
        Ua=U_loc(N1,N2,i)
        Wnu=Wn(N1,N2,u,i)
        err.append(abs(Ua-Wnu))
    #print("The err list: ",err)
    error=max(err)
    return error
def U_loc(N1,N2,n_e):
    Nlb=6
    x=np.array([0]*Nlb,dtype=float)
    y=np.array([0]*Nlb,dtype=float)
    index_lis=Tb_function(N1,N2,n_e)
    for i in range(Nlb):
        x[i],y[i]=Pb_function(N1,N2,index_lis[i])
    x0=(x[0]+x[1]+x[2])/3
    y0=(y[0]+y[1]+y[2])/3
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
    Nlb=6
    x=np.array([0]*Nlb,dtype=float)
    y=np.array([0]*Nlb,dtype=float)
    ub=[0]*Nlb
    index_lis=Tb_function(N1,N2,n_e)
    for i in range(Nlb):
        x[i],y[i]=Pb_function(N1,N2,index_lis[i])
        ub[i]=u[index_lis[i]]
    xm=(x[0]+x[1]+x[2])/3
    ym=(y[0]+y[1]+y[2])/3
    # xm=0.5*(x[0]+x[1])
    # ym=0.5*(y[0]+y[1])
    J=(x[1]-x[0])*(y[2]-y[0])-(x[2]-x[0])*(y[1]-y[0])
    x_t=(1/J)*((y[2]-y[0])*(xm-x[0])-(x[2]-x[0])*(ym-y[0]))
    y_t=(1/J)*(-(y[1]-y[0])*(xm-x[0])+(x[1]-x[0])*(ym-y[0]))
    W=ub[0]*pusi(x_t,y_t,0)+ub[1]*pusi(x_t,y_t,1)+ub[2]*pusi(x_t,y_t,2)+\
      ub[3]*pusi(x_t,y_t,3)+ub[4]*pusi(x_t,y_t,4)+ub[5]*pusi(x_t,y_t,5)
    return W
def U_analy(x,y):
    return x*y*(1-0.5*x)*(1-y)*np.exp(x+y)

i=16
N1=i
N2=i
A=get_stiffness_mat(N1,N2)
b=get_load_vector(N1,N2)
# print("A is: ",'\n',A)
# print("Det(A) is: ",np.linalg.det(A))
u=np.dot(np.linalg.inv(A),b)
print("u is: ",'\n',u)
err=get_err(N1,N2,u)
print("The err is: ",err)
# # index_lis=Tb_function(N1,N2,3)
# # x,y=Pb_function(N1,N2,4)
# # print("The index_lis is: ",index_lis)
# # print("x, y: ",x,y)
