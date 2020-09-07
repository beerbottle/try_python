##调用库的方法
import numpy as np
import matplotlib.pyplot as plt 
import sympy

def f(x):
    return x**3-5*x**2+9

print(f(4))

print(f(1))

#绘图
x=np.linspace(-5,5,num=100)
y=f(x)
plt.plot(x,y)
plt.show()
def exp(x):
    return np.e**x

def exp2(x):
    sum=0
    for k in range(100):
        sum+=float(x**k)/np.math.factorial(k)
    return sum

x=np.linspace(0,10,endpoint=False)
y1=np.log2(x)
y2=np.log(x)
y3=np.log10(x)
plt.plot(x,y1,'red',x,y2,'yellow',x,y3,'blue')
plt.show()

#Trigonometric Functions
plt.plot(np.linspace(-2*np.pi,2*np.pi),np.sin(np.linspace(-2*np.pi,2*np.pi)))
plt.show()

#复核函数
def f(x):
    return x+1
def g(x):
    return x**2
def h(x):
    return f(g(x))
x=np.array(range(-10,10))
y=np.array([h(i) for i in x])

h2=lambda x: f(g(x))
plt.plot(x,h2(x),'rs')
plt.show()

#高阶函数
def horizonntal_shift(f,H):
    return lambda x:f(x-H)
x=np.linspace(-10,10,100)
shifted_g=horizonntal_shift(g,2)
plt.plot(x,g(x),'b',x,shifted_g(x),'r')
plt.show()

#虚数
x=np.linspace(-np.pi,np.pi)
lhs=np.e**(1j*x)
rhs=np.cos(x)+1j*np.sin(x)
print(sum(lhs==rhs)==len(x))

for p in np.e**(1j*x):
    plt.polar([0,np.angle(p)],[0,abs(p)],marker='0')
plt.show()

#limit
import sympy
x=sympy.Symbol('x')
exp=np.e**x 
summs=0
for i in range(20):
    numerator=exp.diff(x,i)
    numerator=numerator.evalf(subs={x:0})
    denominator=np.math.factorial(i)
    summs+=numerator/denominator*x**i
rint(exp.evalf(subs={x:0})-summs.evalf(subs={x:0})


#limit 
x=sympy.Symbol('x',real=True)
f=lambda x:x**2-2*x-6
y=f(x)
print(y.limit(x,2))