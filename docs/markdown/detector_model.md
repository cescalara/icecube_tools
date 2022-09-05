---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3 (ipykernel)
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
from icecube_tools.detector.detector import IceCube, TimeDependentIceCube
```

The `IceCubeData` class can be used for a quick check of the available datasets on the IceCube website. 

```python
my_data = IceCubeData()
my_data.datasets
```

<!-- #region -->
## Effective area, angular resolution and energy resolution of 10 year data

We can now use the date string to identify certain datasets. Let's say we want to use the effective area and angular resolution from the `20210126` dataset. If you don't already have the dataset downloaded, `icecube_tools` will do this for you automatically. This 10 year data release provides more detailed IRF data.

We restrict our examples to the Northern hemisphere.

The simpler, earlier versions are explained afterwards.

The format of the effective area has not changed, though.


For this latest data set, we have different detector configurations available, as they changed through time (the detector was expanded). We can invoke a chosen configuration throught the second argument in `EffectiveArea.from_dataset()` for this particular data set only.
<!-- #endregion -->

```python
from icecube_tools.detector.r2021 import R2021IRF
from icecube_tools.detector.effective_area import EffectiveArea
```

```python
my_aeff = EffectiveArea.from_dataset("20210126", "IC86_II")
```

```python
fig, ax = plt.subplots()
h = ax.pcolor(my_aeff.true_energy_bins, my_aeff.cos_zenith_bins,
            my_aeff.values.T, norm=LogNorm())
cbar = fig.colorbar(h)
ax.set_xscale("log")
ax.set_xlim(1e2, 1e9)
ax.set_xlabel("True energy [GeV]")
ax.set_ylabel("cos(zenith)")
cbar.set_label("Aeff [m^2]")
```

### Energy resolution

Angular resolution depends on the energy resolution. The [paper](https://arxiv.org/pdf/2101.09836.pdf) accompaying the data release explains the dependency: For each bin of true energy and declination a certain amount of events is simulated. These are sorted first into bins of reconstructed energy. These are then reconstructed in terms of `PSF`(the kinematic angle between the incoming neutrino and the outgoing muon after a collision) and actual angular error. Data is given as fractional counts in the bin $(E_\mathrm{reco}, \mathrm{PSF}, \mathrm{ang\_err})$ of all counts in bin (E_\mathrm{true}, \delta). This is nothing but a histogram, corresponding to a probability of finding an event with given true energy and true declination: $p(E_\mathrm{reco}, \mathrm{PSF}, \mathrm{ang\_err} \vert E_\mathrm{true}, \delta)$.

We find the energy resolution, i.e. $p(E_\mathrm{reco} \vert E_\mathrm{true}, \delta)$, by summing over (marginalising over) all entries of $\mathrm{PSF}, \mathrm{ang\_err}$ for the reconstructed energy we are interested in.

The `R2021IRF()`class is able to do so:

```python
irf = R2021IRF.from_period("IC86_II")
```

```python
fig, ax = plt.subplots()
idx = [0, 3, 6, 9, 12]
#plotting Ereco for different true energy bins, the declination bin here is always from +10 to +90 degrees.
for i in idx:
    
    x = np.linspace(*irf.reco_energy[i, 2].support(), num=1000)
    ax.plot(x, irf.reco_energy[i, 2].pdf(x), label=irf.true_energy_bins[i])
ax.legend()
```

This should look like the Fig.4, left panel, of the mentioned paper, the y-axis is only scaled by a constant factor, corresponding to a properly normalised distribution. On this topic, it should be mentioned that the quantities distributed according to these histograms are the logarithms of reconstructed energy, PSF and angular uncertainty! Accordingly, logarithmic quantities are drawn as samples and only exponentiated for calculations and final data products.


### Etrue vs. Ereco

Below, a colormap of the conditional probability $P(E_\mathrm{reco} \vert E_\mathrm{true})$ is shown. It is normalised for each Etrue bin.

```python
etrue = irf.true_energy_bins
ereco = np.linspace(1, 8, num=100)
```

```python
vals = np.zeros((etrue.size, ereco.size))
for c, et in enumerate(etrue[:-1]):
    vals[c, :-1] = irf.reco_energy[c, 2].pdf(ereco[:-1])
```

```python
fig, ax = plt.subplots()
h = ax.pcolor(np.power(10, etrue), np.power(10, ereco), vals.T, norm=LogNorm())
cbar = fig.colorbar(h)
ax.set_xlim(1e2, 1e9)
ax.set_ylim(1e1, 1e8)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("True energy [GeV]")
ax.set_ylabel("Reconstructed energy [GeV]")
cbar.set_label("P(Ereco|Etrue)")
```

### Angular resolution

Now that we have the reconstructed energy for an event with some $E_\mathrm{true}, \delta$, we can proceed in finding the angular resolution.

First, from the given "history" of the event, we sample the matching distribution/histogram of $\mathrm{PSF}$, again my marginalising over the uninteresting quantities, in this case only $\mathrm{ang\_err}$. We thus sample a value of $\mathrm{PSF}$, to whom a distribution of $\mathrm{ang\_err}$ belongs, which we subsequently sample. This is now to be understood as a cone of a given angular radius, within which the true arrival direction lies with a probability of 50%.

For both steps, the histograms are created by `R2021IRF()` when instructed to do so: we pass a tuple of vectors (ra, dec) in radians and a vector of $\log_{10}(E_\mathrm{true})$ to the method `sample()`. Returned are sampled ra, dec (both in radians), angular error (68%, in degrees) and reconstructed energy in GeV.

```python
irf.sample((np.full(4, np.pi), np.full(4, np.pi/4)), np.full(4, 2))
```

If you are interested in the actual distributions, they are accessible through the attributes `R2021IRF().reco_energy` (2d numpy array storing `scipy.stats.rv_histogram` instances) and `R2021IRF().maginal_pdf_psf` (a modified ditionary class instance, indexed by a chain of keys):

```python
etrue_bin = 0
dec_bin = 2
ereco_bin = 10
print(irf.marginal_pdf_psf(etrue_bin, dec_bin, ereco_bin, 'bins'))
print(irf.marginal_pdf_psf(etrue_bin, dec_bin, ereco_bin, 'pdf'))
```

The same works for `marginal_pdf_angerr`. The entries are only created once the `sample()` method needs to. See the  class defintion of `ddict` in `icecube_tools.utils.data` for more details.


### Mean angular uncertainty

We can find the mean angular uncertainty by sampling a large amount of events for different true energies, assuming that the average is sensibly defined as the average of logarithmic quantities.

```python
num = 10000
loge = irf.true_energy_bins
mean_uncertainty = np.zeros(loge.shape)
for c, e in enumerate(loge[:-1]):
    _, _, samples, _ = irf.sample((np.full(num, np.pi), np.full(num, np.pi/4)), np.full(num, e))
    mean_uncertainty[c] = np.power(10,np.average(np.log10(samples)))
mean_uncertainty[-1] = mean_uncertainty[-2]
plt.step(loge, mean_uncertainty, where='post')
```

## Constructing a detector

A detector used for e.g. simulations can be constructed from angular/energy uncertainties and an effective area:

```python
detector = IceCube(my_aeff, irf, irf, "IC86_II")
```

`irf = R2021IRF()` is used both as spatial and energy resolution, because it encompasses both types of information and inherits from both classes. 


## Time dependent detector

We can construct a "meta-detector" spanning multiple data periods through the class `TimeDependentIceCube` from strings defining the data periods. Alternatively, a 

```python
tic = TimeDependentIceCube.from_periods("IC86_I", "IC86_II")
tic.detectors
```

Available periods are

```python
TimeDependentIceCube._available_periods
```

# Effective area, angular resolution and energy resolution of earlier releases

Repeating the prodecure for the `20181018` dataset.

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

```python

```
