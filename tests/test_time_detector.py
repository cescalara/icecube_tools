from logging import raiseExceptions
from icecube_tools.detector.detector import TimeDependentIceCube
from icecube_tools.detector.effective_area import EffectiveArea
from icecube_tools.detector.r2021 import R2021IRF
from pytest import raises


period = "IC86_II"

def test_time_dependent_icecube():

    tic = TimeDependentIceCube.from_periods("IC86_II")
    
    with raises(ValueError):
        tic = TimeDependentIceCube.from_periods("this_is_not_a_period")


def test_time_dependent_aeff():

    aeff = EffectiveArea.from_dataset("20210126", period)
    assert period in aeff._filename

    with raises(ValueError):
        aeff = EffectiveArea.from_dataset("20210126", "not_a_period")


def test_time_dependent_irf():

    irf = R2021IRF.from_period("IC86_II")
    
    with raises(ValueError):
        irf = R2021IRF.from_period("not_a_period")

