import numpy as np
import matplotlib.pyplot as plt
import numerical_calculus as n_cal
x = 0.25; r = 2.0; nlevels = 5; nvalues = 10
# h = 1.0 / (r**(np.arange(nvalues)))
h= r**(-(np.arange(nvalues)))
f = lambda x: np.sin(np.pi*x)
d2fdx2 = lambda x: -np.pi*np.pi*np.sin(np.pi*x)
error_values, q2_fig = n_cal.extrapolate_derivative(x,h,nvalues,nlevels,r,f,d2fdx2)
print('Q2 test output:')
print(error_values)

x=1.0
f = lambda x: np.maximum(0, (x-1)**3)
d2fdx2 = lambda x: np.maximum(0, 6*(x-1))
error_values, fig = n_cal.extrapolate_derivative(x,h,nvalues,nlevels,r,f,d2fdx2)
print("Maximum question:")
print(error_values)