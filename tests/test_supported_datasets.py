from icecube_tools.detector.effective_area import (
    EffectiveArea,
)
from icecube_tools.detector.angular_resolution import (
    AngularResolution,
)
from icecube_tools.detector.energy_resolution import (
    EnergyResolution,
)
from icecube_tools.detector.r2021 import R2021IRF
from icecube_tools.detector.detector import IceCube


def test_aeff_load():

    for dataset_id in EffectiveArea.supported_datasets:

        _ = EffectiveArea.from_dataset(dataset_id)


def test_angres_load():

    for dataset_id in AngularResolution.supported_datasets:

        _ = AngularResolution.from_dataset(dataset_id)


def test_eres_load():

    for dataset_id in EnergyResolution.supported_datasets:

        _ = EnergyResolution.from_dataset(dataset_id)


def test_2021irf_load():

    _ = R2021IRF()


def test_icecube_load():

    aeff = EffectiveArea.from_dataset(EffectiveArea.supported_datasets[0])

    eres = EnergyResolution.from_dataset(EnergyResolution.supported_datasets[0])

    ares = AngularResolution.from_dataset(AngularResolution.supported_datasets[0])

    detector = IceCube(aeff, eres, ares)
