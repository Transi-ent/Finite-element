num=int(input("输入猴子的数目:"))
def fn(n):
    if n==num:
        return(4*x)       #最后剩的桃子的数目
    else:
        return(fn(n+1)*5/4+1)
    
x=1
while 1:
    count=0
    for i in range(1,num):
        if fn(i)%4==0 :
            count=count+1
    if count==num-1:
        print("海滩上原来最少有%d个桃子" % int(fn(0)))
        break
    else:
        x=x+1
