### Import modules ###
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
### No further imports should be required ###



### Functions provided for the coursework ###

def centred_difference(x,h,f):
    """
    Approximate the second derivative of a function f at a point x using
    the standard centred three-point finite difference approximation.

    Inputs:
    ----------
    x (float) : the value at which to approximate the derivative
    h (float) : the displacement to use in the derivative approximation
    f (function) : the function to be differentiated

    Outputs:
    ----------
    derivative_approximation (float) : the approximation to the derivative
    """

    # Construct the standard centred difference approximation of the
    # second derivative.
    derivative_approximation = (f(x+h)-2*f(x)+f(x-h))/(h*h)

    return derivative_approximation


def trapezoidal_integration(a,b,f):
    """
    Approximate the definite integral of a function f over an interval
    [a,b] using the trapezoidal rule.

    Inputs:
    ----------
    a (float) : the lower limit of integration
    b (float) : the upper limit of integration
    f (function) : the function to be integrated

    Outputs:
    ----------
    integral (float) : the approximation to the integral
    """

    # Compute the trapezoidal rule approximation
    integral = 0.5*(b-a)*(f(a)+f(b))

    return integral

### End of provided functions



### Functions to be edited for the coursework ###

### QUESTION 1 - derivative approximation
def approximate_derivative(x: float,h:np.ndarray,nvalues:int,f: Callable[[float],float],d2fdx2: Callable[[float], float],produce_fig:bool):

    """
    Approximates the second derivative of a function f at differing values of 'h' using the centered difference method

    Inputs:
    ----------
    x (float): The point at which we aim to evaluate the second derivative of f at
    h (numpy.ndarray): The range of values of h to use for the centered difference method
    nvalues (integer): The number of approximations to make with a range of h values
    f (Callable): The function of which its second derivative will be approximated
    d2fdx2 (Callable): The precise second derivative of the function 'f'
    produce_fig (boolean): Signals whether we produce a graph of the errors against h

    Outputs:
    ----------
    approx_values (numpy.ndarray): The approximations generated through the centered difference method with shape (nvalues,)
    error_values (numpy.ndarray): The corresponding absolute error values with shape (nvalues,)
    fig (matplotlib.figure): The plot of errors against h. None if 'produce_fig' is False.

    """

    #Populate the array to later fill with approximations
    approx_values = np.zeros(nvalues,)
    error_values = np.zeros(nvalues,)

    #Calculate the approximations for each 'h' value, and calculate the absolute difference between the actual second derivative and its approximate values
    approx_values = centred_difference(x,h,f)
    error_values = np.abs(d2fdx2(x) - approx_values)

    fig=None
    #Create figure plot if produce_fig is True       
    if(produce_fig):
        fig = plt.figure() 
        plt.loglog(h, error_values, 'o-')
        plt.xlabel("h-values")
        plt.ylabel("error values")
        plt.show()

    return approx_values, error_values, fig


### QUESTION 2 - extrapolation
def extrapolate_derivative(x:float,h:np.ndarray,nvalues:int,nlevels:int,r:float,f:Callable[[float],float],d2fdx2:Callable[[float],float]):

    """
    Inputs:
    ----------
    x (float): The value at which the function f will be approximated
    h (np.ndarray): The range of h values for which the extrapolated difference will be used on
    nvalues (integer): The number of values in the array 'h'
    nlevels (integer): The number of extrapolated approximations to generate
    r (float): The factor to which we use to reduce h during refinement
    f (Callable): The function to apply the extrapolation process to
    d2fdx2 (Callable): The exact second derivative of 'f' used to create absolute errors

    Outputs:
    ----------
    error_values (numpy.ndarray): The absolute error values of shape (nvalues, nlevels+1)
    fig (matplotlib.figure): A plot of the errors against 'h' values and the extrapolated approximations

    """

    #Populate the error_values array with zeroes and compute the initial approximation for N_{0}(h)
    error_values = np.zeros((nvalues, nlevels+1), dtype=float)
    current_n, err, _ = approximate_derivative(x,h,nvalues,f,d2fdx2,False)

    for l in range(0, nlevels+1):
        for k in range(0, nvalues-l):
            #Find the absolute error of N_L for each h_k
            error_values[k,l] = np.abs(d2fdx2(x) - current_n[k])

        #Update the extrapolations
        for k in range(0, nvalues - l - 1):
            current_n[k] = ((r**(2*(l+1)))*current_n[k+1] - current_n[k]) / (r**2*(l+1) - 1)


    # Plot of errors against h_{k} values
    fig = plt.figure()
    x_vals = [np.array(h[:nvalues-l]) for l in range(nlevels+1)]
    y_vals = [np.array(error_values[0:nvalues-l, l]) for l in range(nlevels+1)]
    for l in range(0,nlevels + 1):
        plt.loglog(x_vals[l], y_vals[l], label=f"Level {l}")
    plt.xlabel("Step values h_k")
    plt.ylabel("Errors")
    plt.legend()
    plt.show()


    return error_values, fig


# QUESTION 3 - Gauss integration rule
def gauss_integration(a,b,f):

    # Remove the following line when you have completed the code
    integral = None

    return integral


# QUESTION 4 - composite numerical integration
def composite_integration(a,b,npanels,f,f_int,integration_rule):

    # Remove the following two lines when you have completed the code
    integral = None
    error_value = None

    return integral, error_value


# QUESTION 5 - errors in composite numerical integration
def composite_errors(a,b,npanels,f,d2fdx2,d4fdx4,f_int):

    # Remove the following three lines when you have completed the code
    error_values = None
    error_bounds = None
    fig = None

    return error_values, error_bounds, fig


# QUESTION 6 - landing time computation
def compute_time(a,b,Nmax,TOL,hinitial,vterm):

    # Remove the following two lines when you have completed the code
    landing_time = None
    niters = None

    return landing_time, niters

#### Your submission should have no code after this point ####
