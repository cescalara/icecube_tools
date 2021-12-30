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

# Source model

`icecube_tools` has a simple source modelling interface built in that we demonstrate here. 

```python
import numpy as np
from matplotlib import pyplot as plt

from icecube_tools.source.flux_model import PowerLawFlux, BrokenPowerLawFlux
from icecube_tools.source.source_model import PointSource, DiffuseSource
```

## Spectral shape


We start by defining a spectral shape, such as a power law or broken power law. Let's start with the definition of a simple power law flux.

```python
# Parameters of power law flux
flux_norm = 1e-18 # Flux normalisation in units of GeV^-1 cm^-2 s^-1 (sr^-1)
norm_energy = 1e5 # Energy of normalisation in units of GeV
spectral_index = 2.0 # Assumed negative slope
min_energy = 1e4 # GeV
max_energy = 1e8 # GeV

# Instantiate
power_law = PowerLawFlux(flux_norm, norm_energy, spectral_index, 
                         min_energy, max_energy)
```

```python
energies = np.geomspace(min_energy, max_energy)

fig, ax = plt.subplots()
ax.plot(energies, [power_law.spectrum(e) for e in energies])
ax.axhline(flux_norm, color="k", linestyle=":")
ax.axvline(norm_energy, color="k", linestyle=":")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("E [GeV]")
ax.set_ylabel("F [GeV^-1 cm^-2 s^-1 (sr^-1)]");
```

We can also use the `PowerLawFlux` class to perform some simple calculations, such as integration of the flux.

```python
total_flux = power_law.integrated_spectrum(min_energy, max_energy) # cm^-2 s^-1 (sr^-1)
total_flux
```

```python
total_energy_flux = power_law.total_flux_density() # GeV cm^-2 s^-1 (sr^-1)
total_energy_flux
```

Sampling from the power law shape is also possible:

```python
samples = power_law.sample(1000)

fig, ax = plt.subplots()
ax.hist(samples, bins=energies, density=True, label="Samples");
ax.plot(energies, [power_law.spectrum(e)/total_flux for e in energies], label="Model")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("E [GeV]")
ax.legend();
```

The `BrokenPowerLaw` class is also available and behaves in a very similar way.


## Diffuse and point sources

Once the spectral shape is defined, we can specify either a `DiffuseSource` or a `PointSource`. It is assumed that diffuse sources are isotropic and the flux model describes the per-steradian flux over the entire $4\pi$ sky. We also specify a redshift of the source such that adiabatic neutrino energy losses can be accounted for. Naturally, `PointSource` objects also have a direction specified in (ra, dec) coordinates.

```python
diffuse_source = DiffuseSource(power_law, z=0.0)

ra = np.deg2rad(50)
dec = np.deg2rad(-10)
point_source = PointSource(power_law, z=0.5, coord=(ra, dec))
```

The original flux model can now be accessed from within the source along with its other properties:

```python
diffuse_source.flux_model
```

Sources and lists of sources can be used as input to simulations, as demonstrated in the simulation notebook.

```python

```
