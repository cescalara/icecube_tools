import numpy as np
from scipy.stats import bernoulli

"""
Functions to sample from a broken bounded power law.
Copied from https://github.com/grburgess/brokenpl_sample/blob/master/sample_broken_power_law.ipynb
"""


def integrate_pl(x0, x1, x2, a1, a2):
    """
    Integrate a broken power law.
    :param x0: Lower limit
    :param x1: Break
    :param x2: Upper limit
    :param a1: Lower index
    :param a2: Upper index
    :return: relative weights of each part (below and above the break) and total integral
    """
    
    # compute the integral of each piece analytically
    int_1 = (np.power(x1, a1 + 1.) - np.power(x0, a1 + 1.)) / (a1 + 1)
    int_2 = np.power(x1, a1 - a2) * (np.power(x2, a2 + 1.) - np.power(x1, a2 + 1.)) / (a2 + 1)
    
    # compute the total integral
    total = int_1 + int_2
    
    # compute the weights of each piece of the function
    w1 = int_1 / total
    w2 = int_2 / total
    
    return w1, w2, total
    
    
def bpl(x, x0, x1, x2, a1, a2):
    """
    Evaluate broken power law
    :param x: Points at which to evaluate
    :param x0: Lower limit
    :param x1: Break
    :param x2: Upper limit
    :param a1: Lower index
    :param a2: Upper index
    :return: Evaluated power law, casts to np.ndarray
    """
    
    # creatre a holder for the values
    out = np.empty_like(x)
    
    # get the total integral to compute the normalization
    _ , _, C = integrate_pl(x0, x1, x2, a1, a2)
    norm = 1. / C
    
    # create an index to select each piece of the function
    idx = x < x1
    
    # compute the lower power law
    out[idx] = np.power(x[idx], a1)
    
    # compute the upper power law
    out[~idx] = np.power(x[~idx], a2) * np.power(x1, a1 - a2)
    
    return out * norm


def sample_bpl(u, x0, x1, x2, a1, a2):
    """
    :param u: Uniform samples $\el [0, 1]$
    :param x0: Lower limit
    :param x1: Break
    :param x2: Upper limit
    :param a1: Lower index
    :param a2: Upper index
    """

    # compute the weights with our integral function
    w1, w2, _ = integrate_pl(x0, x1, x2, a1, a2)

    # create a holder array for our output
    out = np.empty_like(u)

    # compute the bernoulli trials for lower piece of the function
    # *if we wanted to do the upper part... we just reverse our index*
    # We also compute these to bools for numpy array selection
    idx = bernoulli.rvs(w1, size=len(u)).astype(bool)

    # inverse transform sample the lower part for the "successes"
    out[idx] = np.power(
        u[idx] * (np.power(x1, a1 + 1.0) - np.power(x0, a1 + 1.0))
        + np.power(x0, a1 + 1.0),
        1.0 / (1 + a1),
    )
    
    # inverse transform sample the upper part for the "failures"
    out[~idx] = np.power(
        u[~idx] * (np.power(x2, a2 + 1.0) - np.power(x1, a2 + 1.0))
        + np.power(x1, a2 + 1.0),
        1.0 / (1 + a2),
    )
    
    return out
