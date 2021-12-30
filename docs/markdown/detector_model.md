---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
# Detector model interface

In addition to the neutrino data itself, the IceCube collaboration provides some information about the detector that can be useful to construct simple simulations and fits. For example, the effective area is needed to connect between incident neutrino fluxes and expected number of events in the detector. 


`icecube_tools` also provides an quick interface to loading and working with such information. This is a work in progress and only certain datasets are currently implemented, such as the ones demonstrated below. 
<!-- #endregion -->

```python
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from icecube_tools.utils.data import IceCubeData
```

The `IceCubeData` class can be used for a quick check of the available datasets on the IceCube website. 

```python
my_data = IceCubeData()
my_data.datasets
```

## Effective area, angular resolution and energy resolution

We can now use the date string to identify certain datasets. Let's say we want to use the effective area and angular resolution from the `20181018` dataset. If you don't already have the dataset downloaded, `icecube_tools` will do this for you automatically.

```python
from icecube_tools.detector.effective_area import EffectiveArea
from icecube_tools.detector.energy_resolution import EnergyResolution
from icecube_tools.detector.angular_resolution import AngularResolution
```

```python
my_aeff = EffectiveArea.from_dataset("20181018")
my_angres = AngularResolution.from_dataset("20181018")
```

```python
fig, ax = plt.subplots()
h = ax.pcolor(my_aeff.true_energy_bins, my_aeff.cos_zenith_bins,
            my_aeff.values.T, norm=LogNorm())
cbar = fig.colorbar(h)
ax.set_xscale("log")
ax.set_xlabel("True energy [GeV]")
ax.set_ylabel("cos(zenith)")
cbar.set_label("Aeff [m^2]")
```

```python
fig, ax = plt.subplots()
ax.plot(my_angres.true_energy_values, my_angres.values)
ax.set_xscale("log")
ax.set_xlabel("True energy [GeV]")
ax.set_ylabel("Mean angular error [deg]")
```

We can also easily check what datasets are supported by the different detector information classes:

```python
EffectiveArea.supported_datasets
```

```python
AngularResolution.supported_datasets
```

```python
EnergyResolution.supported_datasets
```

<!-- #region -->
If you would like to see some other datasets supported, please feel free to open an issue or contribute your own!


For the `20150820` dataset, for which we also have the energy resolution available...
<!-- #endregion -->

```python
my_aeff = EffectiveArea.from_dataset("20150820")
my_eres = EnergyResolution.from_dataset("20150820")
```

```python
fig, ax = plt.subplots()
h = ax.pcolor(my_aeff.true_energy_bins, my_aeff.cos_zenith_bins,
            my_aeff.values.T, norm=LogNorm())
cbar = fig.colorbar(h)
ax.set_xscale("log")
ax.set_xlabel("True energy [GeV]")
ax.set_ylabel("cos(zenith)")
cbar.set_label("Aeff [m^2]")
```

```python
fig, ax = plt.subplots()
h = ax.pcolor(my_eres.true_energy_bins, my_eres.reco_energy_bins, my_eres.values.T, 
              norm=LogNorm())
cbar = fig.colorbar(h)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("True energy [GeV]")
ax.set_ylabel("Reconstructed energy [GeV]")
cbar.set_label("P(Ereco|Etrue)")
```

## Detector model

We can bring together these properties to make a detector model that can be used for simulations. 

```python
from icecube_tools.detector.detector import IceCube
```

```python
my_detector = IceCube(my_aeff, my_eres, my_angres)
```
