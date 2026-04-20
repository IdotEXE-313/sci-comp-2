### Import modules ###
import numpy as np
import scipy as sp
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
        plt.title("Error Values Against H-Values")
        plt.show()

    return approx_values, error_values, fig


### QUESTION 2 - extrapolation
def extrapolate_derivative(x:float,h:np.ndarray,nvalues:int,nlevels:int,r:float,f:Callable[[float],float],d2fdx2:Callable[[float],float]):

    """
    Needs explanation


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
            current_n[k] = ((r**(2*(l+1)))*current_n[k+1] - current_n[k]) / (r**(2*(l+1)) - 1)


    # Plot of errors against h_{k} values
    fig = plt.figure()
    x_vals = [np.array(h[:nvalues-l]) for l in range(nlevels+1)]
    y_vals = [np.array(error_values[0:nvalues-l, l]) for l in range(nlevels+1)]
    for l in range(0,nlevels + 1):
        plt.loglog(x_vals[l], y_vals[l],"-o", label=f"Level {l}")
    plt.xlabel("Step values h_k")
    plt.ylabel("Errors")
    plt.title("Plot Of Errors Against 'H' Values and Extrapolated Approximations")
    plt.legend()
    plt.show()


    return error_values, fig


# QUESTION 3 - Gauss integration rule
def gauss_integration(a:float,b:float,f:Callable[[float],float]):

    """
    Returns an approximation to the integral of 'f' over an interval, [a,b], using the 2 point Gauss integration rule

    Inputs: 
    ----------
    a (float): The lower value of the interval [a,b]
    b (float): The upper value of the interval [a,b]
    f (Callable): The function to apply the Gauss integration rule to

    Outputs:
    ----------
    integral (float): The approximation to the integral of 'f' over the interval [a,b] using the Gauss integration rule.
    """

    #Define the weights and nodes of the points provided in the coursework instructions
    weights = np.array([1.0,1.0])
    nodes = np.array([-1.0/np.sqrt(3.0),1/np.sqrt(3.0)])

    #Transform these weights and nodes to account for the usual assumption of being on the domain [-1,1]
    shifted_weights = (0.5 * (b-a)) * weights
    shifted_nodes = 0.5 * ((b-a)*nodes + (b+a))

    integral = np.sum(shifted_weights * f(shifted_nodes))
    return integral


# QUESTION 4 - composite numerical integration
def composite_integration(a:float,b:float,npanels:int,f:Callable[[float],float],f_int:Callable[[float],float],integration_rule:Callable[[float, float, Callable[[float],float]],float]):

    """
    Computes an approximation of the integral of 'f' over the interval [a,b] using the composite integration method

    Inputs:
    ----------
    a (float): The lower value of the interval [a,b]
    b (float): The upper value of the interval [a,b]
    npanels (integer): The number of panels that the interval [a,b] is divided into
    f (Callable): The function that we will apply composite integration to
    f_int (Callable): The indefinite integral of 'f'
    integration_rule (Callable): The numerical method of integration to be applied (either trapezium or gauss integration)

    Outputs:
    ----------
    integral (float): The approximation to the integral of 'f' after applying composite integration
    error_value (float): The absolute value of the error between the exact integral and the approximation over [a,b]
    """

    #Generate the upper limit of each subinterval 
    upper_lims = np.linspace(a,b,npanels+1)
    approximations = np.zeros(npanels,)

    #Apply the integration rule to each subinterval
    for endpoint in range(0,len(upper_lims)-1):
        approximations[endpoint] = integration_rule(upper_lims[endpoint], upper_lims[endpoint + 1], f)
    integral = np.sum(approximations)

    #Calculate the errors of the exact integral and the approximate integral
    exact_integral = f_int(b) - f_int(a)
    error_value = np.abs(exact_integral - integral)

    return integral, error_value


# QUESTION 5 - errors in composite numerical integration
def composite_errors(a:float,b:float,npanels:int,f:Callable[[float],float],d2fdx2:Callable[[float],float],d4fdx4:Callable[[float],float],f_int:Callable[[float],float]):

    """
    Needs explanation

    Inputs:
    ----------
    a (float): The lower end of the interval [a,b]
    b (float): The upper end of the interval [a,b]
    npanels (integer): The number of subintervals to create when applying the composite integration function
    f (Callable): The function on which the integral is approximated over [a,b]
    d2fdx2 (Callable): The exact second derivative of 'f', used to compute error bounds for the Trapezium rule
    d4fd4x (Callable): The exact fourth derivative of 'f', used to computer error bounds for the 2-point Gaussian integration rule
    f_int (Callable): The exact integral of 'f', used to calculate error values

    Outputs:
    ----------
    error_values (numpy.ndarray): A numpy array of shape (2,npanels.size) with the error values of the Trapezium rule on [0,k], and error values of Gaussian integration on [1,k]
    error_bounds (numpy.ndarray): A numpy array of shape (2,npanels.size) with the error bounds of the Trapezium rule on [0,k], and error bounds of Gaussian integration on [1,k]
    fig (matplotlib.fig): A plot showing the error bounds and values against the number of panels used in each integration method
    """

    
    error_values = np.zeros((2, npanels.size))
    error_bounds = np.zeros((2, npanels.size))
    upper_lims = np.linspace(a,b,1001) #Split the interval [a,b] into 1000 equal sized subintervals, used to generate the error bounds
    midpoints = 0.5 * (upper_lims[:-1] + upper_lims[1:])

    max_second = np.max(np.abs(d2fdx2(midpoints)))
    max_fourth = np.max(np.abs(d4fdx4(midpoints)))

    # Store the errors for each panel size (computed in the composite integration function)
    for k,panel in enumerate(npanels):
        h = (b-a)/panel
        trap_approx, trap_err = composite_integration(a,b,panel,f,f_int,trapezoidal_integration)
        gauss_approx, gauss_err = composite_integration(a,b,panel,f,f_int,gauss_integration)

        error_values[0,k] = trap_err
        error_values[1,k] = gauss_err

        # Calculate the error bounds for each integration method, and then store them
        error_bounds[0,k] = ((h**2)/12) * (b-a) * max_second
        error_bounds[1,k]= ((h**4)/4320) * (b-a) * max_fourth

    # Set up the required graph and show it
    fig = plt.figure()
    plt.loglog(npanels, error_values[0, :], label="Error Values (Trapezium Rule)")
    plt.loglog(npanels, error_bounds[0,:], label="Error Bounds (Trapezium Rule)")
    plt.loglog(npanels, error_values[1,:], label="Error Values (Gaussian)")
    plt.loglog(npanels, error_bounds[1,:], label="Error Bounds (Gaussian)")

    plt.xlabel("Number of Panels")
    plt.ylabel("Error Values/Bounds")
    plt.title("Error Bounds/Values Against Number Of Panels Used In Each Integration Method")
    plt.legend()
    plt.show()


    return error_values, error_bounds, fig


# QUESTION 6 - landing time computation
def compute_time(a:float,b:float,Nmax:int,TOL:float,hinitial:float,vterm:float):

    """
    Given the physical situation (constant speed, object falling to the ground), then we know that we can estimate an initial guess for a root of F(T) as time = distance / speed.
    Given also that the derivative of F(T) is analytical (it is just -v(t)), and -v(200) = -5 (which is sufficiently far from 0, i.e. F'(T) is not equal to zero at the root) and is 
    twice differentiable in the domain [1,2000], then we can apply the Newton-Rasphon method for our nonlinear solver. We choose this because, under the correct conditions (which we have),
    Newton's method converges quadratically. It may have been sufficient to use the bisection method, but this converges linearly, so the overall iterations (and thus v-iterations)
    is signficiantly lower when using the Newton method. 
    We choose Gaussian integration because, per evaluation, Gaussian two-point integration provides much greater accuracy by selecting weighted nodes to capture curvature that would
    require a larger number of panels to accurately approximate under the trapezoidal rule. Hence, the overall number of panels is much smaller under Gaussian two-point integration.
    Finally, we select 445 panels for the two-point Gaussian integration as an iterative process; at 1000 panels, we get 3000 viters, and at 500 panels we get 1500 viters (both provide
    200.353 as the root to 3 decimal places, so we work with this as the assumption of the correct answer); at 440 panels, we see stability towards 200.353, and we add 5 panels for added
    stability which provides us with 1335 viters.

    Inputs:
    ----------
    a (float): The lower end of the interval [a,b]
    b (float): The upper end of the interval [a,b]
    Nmax (integer): The maximum number of iterations to run in our nonlinear solver
    TOL (float): The value at which we terminate the nonlinear solver before reaching Nmax
    hinitial (float): The initial height that the jumper is at above ground level
    vterm (float): The terminal velocity the jumper reaches whilst falling

    Outputs:
    ----------
    landing_time (float): An approximation to the time it took the jumper to reach ground level
    niters (integer): The number of iterations it took for the nonlinear solver to find a root of 'f'. This will be equal to Nmax if the convergence criteria isn't met.
    
    """

    viters = [0]
    initial_guess = hinitial / vterm #Approximate the initial guess of the time taken to fall to ground with time = distance / speed

    # Define the function of velocity (dx/dt) as stated in the assignment notes
    g = 9.81
    def v(t):
        viters[0] += 1
        return vterm * np.tanh((g*t)/vterm)
    f_int = lambda x: 0
    f_prime = lambda t: -vterm * np.tanh((g*t)/vterm) # We have that the derivative of F(T) is just -v(t)
    f = lambda t: hinitial - composite_integration(0,t,445,v,f_int,gauss_integration)[0]

    root, info = sp.optimize.newton(f, initial_guess, f_prime, maxiter=Nmax, tol=TOL, full_output=True)
    return root, info.iterations
    

#### Your submission should have no code after this point ####
