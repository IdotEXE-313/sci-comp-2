import numpy as np
import matplotlib.pyplot as plt
import numerical_calculus as n_cal
a = 1.0; b = 2.0
npanels = 100 * 2**(np.arange(11))
f = lambda x: np.sin(np.pi*x)
d2fdx2 = lambda x: -np.pi*np.pi*np.sin(np.pi*x)
d4fdx4 = lambda x: np.pi*np.pi*np.pi*np.pi*np.sin(np.pi*x)
f_int = lambda x: -np.cos(np.pi*x)/np.pi
error_values, error_bounds, q5_fig_f1 = n_cal.composite_errors(a,b,npanels,f,d2fdx2,d4fdx4,f_int)
print('Q5 test output for sin(pi*x):')
print(error_values)
print(error_bounds)

a = 0.0; b = 1.0
npanels = 100 * 2**(np.arange(11))
f = lambda x: x*np.log(np.maximum(x,1e-10))
d2fdx2 = lambda x: 1.0/x
d4fdx4 = lambda x: 2.0/(x*x*x)
f_int = lambda x: x*x*np.log(np.maximum(x,1e-10))/2.0 - x*x/4.0
error_values, error_bounds, q5_fig_f2 = n_cal.composite_errors(a,b,npanels,f,d2fdx2,d4fdx4,f_int)
print('Q5 test output for x*ln(x):')
print(error_values)
print(error_bounds)