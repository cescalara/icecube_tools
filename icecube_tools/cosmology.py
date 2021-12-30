import numpy as np

Om = 0.3
Ol = 0.7
H0 = 70  # km s^-1 Mpc^-1
c = 3e5  # km s^-1
DH = c / H0  # Mpc

Mpc_to_cm = 3.086e24
m_to_cm = 100
yr_to_s = 3.154e7


def xx(z):
    """
    Helper function for the computation of
    :py:func:`icecube_tools.cosmology.luminosity_distance`.
    """

    return ((1 - Om) / Om) / pow(1 + z, 3)


def phi(x):
    """
    Helper function for the computation of
    :py:func:`icecube_tools.cosmology.luminosity_distance`.
    """

    x2 = np.power(x, 2)
    x3 = pow(x, 3)
    numerator = 1.0 + (1.320 * x) + (0.4415 * x2) + (0.02656 * x3)
    denominator = 1.0 + (1.392 * x) + (0.5121 * x2) + (0.03944 * x3)

    return numerator / denominator


def luminosity_distance(z):
    """
    Luminosity distance based on approximation used in Adachi & Kasai 2012.

    :param z: Redshift
    :type z: float
    :return: Luminosity distance in Mpc
    :rtype: float
    """

    x = xx(z)
    zp = 1 + z

    A = 2 * DH * zp / np.sqrt(Om)
    B = phi(xx(0)) - ((1 / np.sqrt(zp)) * phi(x))

    return A * B


def comoving_distance(z):
    """
    Comoving distance.

    :param z: Redshift
    :type z: float
    :return: Comoving distance in Mpc
    :rtype float:
    """
    return luminosity_distance(z) / (1 + z)


def E_fac(z):
    """
    E(z) = sqrt(Om * (1+z)^3)

    :param z: Redshift
    :type z: float
    :return: E(z)
    :rtype float:
    """

    Omp = Om * (1 + z) ** 3

    return np.sqrt(Omp + Ol)


def differential_comoving_volume(z):
    """
    Differential comoving volume.

    :param z: Redshift
    :type z: float
    :return: Differential comoving volume in Mpc^3
    :rtype: float
    """

    dc = comoving_distance(z)

    return (DH * dc ** 2) / E_fac(z)
