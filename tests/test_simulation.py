import numpy as np
from pytest import approx

from icecube_tools.detector.effective_area import EffectiveArea
from icecube_tools.detector.energy_resolution import EnergyResolution
from icecube_tools.detector.angular_resolution import AngularResolution
from icecube_tools.detector.detector import IceCube
from icecube_tools.source.flux_model import PowerLawFlux
from icecube_tools.source.source_model import DiffuseSource, PointSource
from icecube_tools.neutrino_calculator import NeutrinoCalculator, PhiSolver
from icecube_tools.simulator import Simulator


aeff = EffectiveArea.from_dataset("20181018")
angres = AngularResolution.from_dataset("20181018")
eres = EnergyResolution.from_dataset("20150820")
detector = IceCube(aeff, eres, angres)

pl_params = (1e-18, 1e5, 2.2, 1e4, 1e8)
pl_flux = PowerLawFlux(*pl_params)
point_source = PointSource(pl_flux, z=0.3)
diffuse_source = DiffuseSource(pl_flux)
sources = [point_source, diffuse_source]


def test_nu_calc():

    nu_calc = NeutrinoCalculator([point_source], aeff)

    Nnu = nu_calc(
        time=1,
        min_energy=1e4,
        max_energy=1e8,
        min_cosz=-1,
        max_cosz=1,
    )[0]

    assert Nnu > 0

    assert ~np.isnan(Nnu)

    phi_solver = PhiSolver(aeff, 1e5, 1e4, 1e8, time=1, min_cosz=-1, max_cosz=1)

    phi_norm = phi_solver(Nnu=Nnu, dec=0.0, index=2.2)

    assert phi_norm == approx(1e-18, rel=0.01)


def test_simulation():

    simulator = Simulator(sources, detector)
    simulator.time = 1

    simulator.run(show_progress=True, seed=42)

    assert min(simulator.true_energy) >= 1e4
