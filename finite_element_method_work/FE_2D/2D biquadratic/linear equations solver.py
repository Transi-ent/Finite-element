"""plot the weekday--hour figure"""
import math
import numpy as np
import datetime
import matplotlib.pyplot as plt
import try1 as t

def f(x,y):
    return a+b*x+c*y+d*x*y+e*x**2+f*y**2+g*x**2*y+h*x*y**2+k+x**2*y**2
A=np.array([[1,-1,-1,1,1,1,-1,-1,1],
            [1,1,-1,-1,1,1,-1,1,1],
            [1,1,1,1,1,1,1,1,1],
            [1,-1,1,-1,1,1,1,-1,1],
            [1,0,-1,0,0,1,0,0,0],
            [1,1,0,0,1,0,0,0,0,],
            [1,0,1,0,0,1,0,0,0,],
            [1,-1,0,0,1,0,0,0,0,],
            [1,0,0,0,0,0,0,0,0,]])
b=np.array([0,0,0,0,0,0,0,0,1])

x=np.linalg.solve(A,b)
#print(A)
print("The coefficient is: ",'\n',x)
