---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Simulation

We can bring together the detector and source modelling to calculate the expected number of neutrino events and run simulations.

```python
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import h5py
```

## Defining a source and detector model

```python
from icecube_tools.detector.effective_area import EffectiveArea
from icecube_tools.detector.energy_resolution import EnergyResolution
from icecube_tools.detector.angular_resolution import AngularResolution
from icecube_tools.detector.detector import IceCube
from icecube_tools.source.flux_model import PowerLawFlux
from icecube_tools.source.source_model import DiffuseSource, PointSource
```

```python
# Define detector (see detector model notebook for more info)
aeff = EffectiveArea.from_dataset("20181018")
angres = AngularResolution.from_dataset("20181018")
eres = EnergyResolution.from_dataset("20150820")
detector = IceCube(aeff, eres, angres)
```

```python
# Define simple sources (see source model notebook for more info)
diff_flux_norm = 1e-18 # Flux normalisation in units of GeV^-1 cm^-2 s^-1 sr^-1
point_flux_norm = 1e-18 # Flux normalisation in units of GeV^-1 cm^-2 s^-1 
norm_energy = 1e5 # Energy of normalisation in units of GeV
min_energy = 1e4 # GeV
max_energy = 1e7 # GeV

diff_power_law = PowerLawFlux(diff_flux_norm, norm_energy, 3.0, 
                              min_energy, max_energy)
diff_source = DiffuseSource(diff_power_law, z=0.0)

point_power_law = PowerLawFlux(point_flux_norm, norm_energy, 2.6, 
                               min_energy, max_energy)
point_source = PointSource(point_power_law, z=0.3, coord=(np.pi, np.pi/4))
sources = [diff_source, point_source]
```

## Expected number of neutrino events

Sometimes we just want to predict the number of events from sources in a detector without specifying all detector properties or running a simulation. We can do this with the `NeutrinoCalculator`. For this, we just need a source list and an effective area.

```python
from icecube_tools.neutrino_calculator import NeutrinoCalculator, PhiSolver
```

```python
nu_calc = NeutrinoCalculator(sources, aeff)
nu_calc(time=1, # years
        min_energy=min_energy, max_energy=max_energy, # energy range
        min_cosz=-1, max_cosz=1) # cos(zenith) range
```

The calculator returns a list of expected neutrino event numbers, one for each source.

We may also want to do the inverse, and find the `PointSource` flux normalisation corresponding to an expected number of events. For this there is the `PhiSolver` class.

```python
phi_solver = PhiSolver(aeff, norm_energy, min_energy, max_energy, 
                       time=1, min_cosz=-1, max_cosz=1)
phi_norm = phi_solver(Nnu=10, 
                      dec=30, # degrees
                      index=2.0) # spectral index
phi_norm # GeV^-1 cm^-2 s^-1
```

## Setup and run simulation

```python
from icecube_tools.simulator import Simulator
```

```python
# Set up simulation
simulator = Simulator(sources, detector)
simulator.time = 1 # year

# Run simulation
simulator.run(show_progress=True, seed=42)
```

This way, the simulator calculates the expected number of neutrinos from these sources given the observation period, effective area and relevant source properties. We note that we could also run a simulation for a fixed number of neutrinos if we want, simply by passing the optional argument `N` to `simulator.run()`.

```python
#simulator.run(N=10, show_progress=True, seed=42)
```

```python
# Save to file
simulator.save("data/sim_output.h5")

# Load
with h5py.File("data/sim_output.h5", "r") as f:
    true_energy = f["true_energy"][()]
    reco_energy = f["reco_energy"][()]
    ra = f["ra"][()]
    dec = f["dec"][()]
    ang_err = f["ang_err"][()]
    source_label = f["source_label"][()]
```

```python
# Plot energies
bins = np.geomspace(1e2, max_energy)
fig, ax = plt.subplots()
ax.hist(true_energy, bins=bins, alpha=0.7, label="E_true")
ax.hist(reco_energy, bins=bins, alpha=0.7, label="E_reco")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("E [GeV]")
ax.legend();
```

```python
# Plot directions
ps_sel = source_label == 1

fig, ax = plt.subplots(subplot_kw={"projection": "aitoff"})
fig.set_size_inches((12, 7))

circles = []
for r, d, a in zip(ra[~ps_sel], dec[~ps_sel], ang_err[~ps_sel]):
    circle = Circle((r-np.pi, d), radius=np.deg2rad(a))
    circles.append(circle)
df_nu = PatchCollection(circles)

circles = []
for r, d, a in zip(ra[ps_sel], dec[ps_sel], ang_err[ps_sel]):
    circle = Circle((r-np.pi, d), radius=np.deg2rad(a))
    circles.append(circle)
ps_nu = PatchCollection(circles, color="r")

ax.add_collection(df_nu)
ax.add_collection(ps_nu)

ax.grid()
```
