import numpy as np


def spherical_to_icrs(theta, phi):
    ra = phi
    dec = np.pi / 2 - theta
    return ra, dec

def icrs_to_spherical(ra, dec):
    phi = ra
    theta = np.pi / 2 - dec
    return theta, phi

def spherical_to_cart(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z