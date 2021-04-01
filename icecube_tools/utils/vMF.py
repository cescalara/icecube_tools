import numpy as np

"""
Conversion between kappa and angular radius.

Based on Equation 11 in Soiaporn et al. 2013.
"""


def get_kappa(theta_p, p=0.68):
    """
    Calculate kappa corresponding to an angular radius theta_p.

    :param theta_p: Angular radius containing probability p [deg]
    :param p: Probability contained within radius theta_p
    :return kappa: Shape parameter of the vMF distribution
    """

    theta_p = np.deg2rad(theta_p)

    return -(2 / theta_p ** 2) * np.log(1 - p)


def get_theta_p(kappa, p=0.68):
    """
    Calculate theta_p corresponding to a vMF distribution
    of width kappa and containing probability p.

    :param kappa: Shape parameter of the vMF distribution
    :param p: Probability contained in radius theta_p
    :return theta_p: Angular radius containing probability p [deg]
    """

    theta_p = np.sqrt((-2 / kappa) * np.log(1 - p))

    return np.rad2deg(theta_p)
