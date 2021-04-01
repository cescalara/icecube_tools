import numpy as np
from icecube_tools.utils.vMF import get_kappa, get_theta_p
from icecube_tools.detector.angular_resolution import AngularResolution


def test_kappa_conversion():

    theta_1sigma = 1.0

    kappa = get_kappa(theta_1sigma, 0.68)

    theta_p = get_theta_p(kappa, 0.68)

    assert theta_1sigma == theta_p


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

    ang_res.sample((ra, dec), Etrue)

    # Return angular error
    assert ang_res.ret_ang_err == ang_res.get_ret_ang_err(Etrue)
