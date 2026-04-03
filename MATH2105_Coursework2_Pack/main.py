import numpy as np
import matplotlib.pyplot as plt
import numerical_calculus as n_cal


# QUESTION 1
x = 0.25; nvalues = 5
h = 0.1 / (2**(np.arange(nvalues)))
f = lambda x: np.sin(np.pi*x)
d2fdx2 = lambda x: -np.pi*np.pi*np.sin(np.pi*x)
approx_values, error_values, q1_fig = n_cal.approximate_derivative(x,h,nvalues,f,d2fdx2,False)
print('Q1 test output:')
print(approx_values)
print(error_values)


# QUESTION 2
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


# QUESTION 3
import numpy as np
import matplotlib.pyplot as plt
import numerical_calculus as n_cal
a = 1.0; b = 2.0
f = lambda x: np.sin(np.pi*x)
integral_gauss = n_cal.gauss_integration(a,b,f)
print('Q3 test output:')
print(integral_gauss)


# QUESTION 4
import numpy as np
import matplotlib.pyplot as plt
import numerical_calculus as n_cal
a = 1.0; b = 2.0; npanels = 10
f = lambda x: np.sin(np.pi*x)
f_int = lambda x: -np.cos(np.pi*x)/np.pi
composite_trapezoidal, error_trapezoidal = n_cal.composite_integration(a,b,npanels,f,f_int,n_cal.trapezoidal_integration)
composite_gauss, error_gauss = n_cal.composite_integration(a,b,npanels,f,f_int,n_cal.gauss_integration)
print('Q4 test output:')
print(composite_trapezoidal,error_trapezoidal)
print(composite_gauss,error_gauss)


# QUESTION 5
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


# QUESTION 6
a = 1.0; b = 2000.0
Nmax = 200; TOL = 1e-10
hinitial = 1000.0; vterm = 5.0
landing_time, niters = n_cal.compute_time(a,b,Nmax,TOL,hinitial,vterm)
print('Q6 test output:')
print(landing_time,niters)
