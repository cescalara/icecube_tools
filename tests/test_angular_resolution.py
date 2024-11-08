import numpy as np
from pytest import approx, raises
from icecube_tools.utils.vMF import get_kappa, get_theta_p
from icecube_tools.detector.angular_resolution import AngularResolution
from icecube_tools.detector.r2021 import R2021IRF

def test_kappa_conversion():

    theta_1sigma = 1.0

    kappa = get_kappa(theta_1sigma, 0.68)

    theta_p = get_theta_p(kappa, 0.68)

    assert theta_1sigma == approx(theta_p)

"""
def test_angular_resolution():

    # Load
    ang_res = AngularResolution.from_dataset(
        "20181018",
        ret_ang_err_p=0.9,
        offset=0.4,
    )

    # Sample
    ra = 0.0  # rad
    dec = np.pi / 4  # rad
    Etrue = 1e5  # GeV

    ang_res.sample(Etrue, (ra, dec))

    # Return angular error
    assert ang_res.ret_ang_err == ang_res.get_ret_ang_err(Etrue)
"""

def test_r2021_irf():

    # Load
    irf = R2021IRF.from_period("IC86_II")

    # Sample
    ra = 0.0 # rad
    dec = np.pi / 4 # rad
    Etrue = np.log10(1e5)

    _, _, _, _ = irf.sample((ra, dec), Etrue)

    with raises(ValueError):
        irf._return_etrue_bins(1, dec)

    with raises(ValueError):
        irf._return_etrue_bins(2.2, np.pi)


