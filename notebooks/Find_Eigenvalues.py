# In this notebook we use symbolic calculation to find all critical points of a function and the eigenvalues associated. This way we can know the shape of the maxima / minima
import sympy as sy
import plotly.graph_objs as go

x,y = sy.symbols('x y')
sy.init_printing(use_unicode=True)

#%% Define Function
#f = x**4+y**2-x*y # function 2 from Stanford
f = 4*x + 2*y - x**2 -3*y**2

df_dx = sy.diff(f,x)
df_dy = sy.diff(f,y)

#%% Find critical points
cr  =sy.nonlinsolve([df_dx,df_dy],[x,y])
print('critical points',cr)
cr
#%% build hessian
e = sy.hessian(f,[x,y])
e

#%% Find eigenvalues for each of the critical points
for c in cr :
    xv = c[0]
    yv = c[1]
    print('Critical point : \n\tx : {} \n\ty : {}'.format(xv.evalf(),yv.evalf()))
    eigs = list(e.subs({x:xv,y:yv}).eigenvals().keys())
    if eigs[0] > 0 and eigs[1] > 0 :
        print('Concave up')
    elif eigs[0] < 0  and eigs[1] < 0 :
        print('Concave down')
    else :
        print('Saddle Point')
    print('Eigen Values : ',eigs)
