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

# Point source likelihood

`icecube_tools` also provides an interface to the likelihoods often used in point source searches of the neutrino data (see [this paper](https://arxiv.org/abs/0801.1604) by Braun et al.).

$$
\mathcal{L} = \prod_{i=1}^N \Bigg[ \frac{n_s}{N} \mathcal{S}(\theta_i, E_i, \gamma) + (1-\frac{n_s}{N}) \mathcal{B}(\theta_i, E_i) \Bigg],
$$
where $N$ is the total number of detected neutrino events, $n_s$ is the expected number of source events, $\theta$ is the neutrino direction, $E$ is the reconstructed neutrino energy and $\gamma$ is the source spectral index.

The point source likelihood is a mixture model with two components: one representing possible astrophysical neutrino sources, $\mathcal{S}(\theta, E)$, and the other known background, $\mathcal{B}(\theta, E)$. Each component has terms depending on the directional or spatial source--neutrino relationship and also the energy of the neutrinos, as higher energy neutrinos are more likely to come from astrophysical sources. Depending on the search, the energy dependence may be omitted. Also, there may be a temporal dependence added, but this is not yet implemented in `icecube_tools`.

Here we implement a simple likelihood and apply it to some simulated data. There are several likelihoods available, and more information can be found in the API documentation.

```python
import numpy as np
from matplotlib import pyplot as plt
import h5py

from icecube_tools.point_source_likelihood.spatial_likelihood import (
    EventDependentSpatialGaussianLikelihood
)
from icecube_tools.point_source_likelihood.energy_likelihood import (
    MarginalisedEnergyLikelihood2021, read_input_from_file,
    MarginalisedIntegratedEnergyLikelihood
)
from icecube_tools.point_source_likelihood.point_source_likelihood import (
    PointSourceLikelihood, TimeDependentPointSourceLikelihood
)

from icecube_tools.detector.effective_area import EffectiveArea
from icecube_tools.detector.r2021 import R2021IRF
```

## Spatial likelihood


We can start with the spatial/directional term. Let's use the energy dependent spatial likelihood. It is build from a Gaussian with an event-wise uncertainty sampled from the IRF data. The background case will simply be an isotropic distribution on the sphere.

```python
angular_resolution = 1 # deg
spatial_likelihood = EventDependentSpatialGaussianLikelihood(angular_resolution)
```

We show the likelihood profile for a single event with an assumed uncertainty of 1 degree.

```python
source_coord = (np.pi, np.deg2rad(30))
test_coords = [(np.pi+_, source_coord[1]) for _ in np.linspace(-0.1, 0.1, 100)]

fig, ax = plt.subplots()
ax.plot(np.rad2deg([tc[0] for tc in test_coords]), 
        [spatial_likelihood(1., tc, source_coord) for tc in test_coords])
ax.axvline(np.rad2deg(source_coord[0]), color="k", linestyle=":", 
           label="Source location")
ax.set_xlabel("RA [deg]")
ax.set_ylabel("Spatial likelihood")
ax.legend();
```

## Energy likelihood


Now let's think about the energy-dependent term. The way this is handled is to marginalise over the true neutrino energies, to directly connect the reconstructed neutrino energies to the spectral index of a simple power-law source model. 

Doing this properly requires a knowledge of the relationship between the true and reconstructed energies as well as the details of the power law model. The most straightforward way to implement this is to simulate the a large number of events using the `Simulator` and build a likelihood using the output of this simulation and `MarginalisedEnergyLikelihoodFromSim`. We do exactly this with pre-computed lists of events, to be found in the data subdirectory: `sim_output_{index}.h5`. These were simulated using point sources with spectral index `index` at 45 degrees declination. The likelihood is restricted to a small band of declination around the assumed source. Using the same declination for our test source, this is fine. For different source declinations further simulations would be needed to account for the declination dependence of the detector acceptance.

```python
aeff = EffectiveArea.from_dataset("20210126", period="IC86_I")
irf = R2021IRF.from_period("IC86_I")
new_reco_bins = irf.reco_energy_bins[12, 2]
energy_likelihood = MarginalisedIntegratedEnergyLikelihood(irf, aeff, new_reco_bins)
#energy_likelihood = MarginalisedEnergyLikelihood2021([1.5, 2.0, 2.5, 3.0, 3.5, 3.7, 4.0], 'data', 'sim_output', np.pi/4,)
# the likelihood class is backwardscompatible with the "older" simulation-based energy likelihood
```

```python
new_reco_bins
```

```python
#test_energies = np.geomspace(10, 1e7) # GeV
test_indices = [2.0, 2.5, 3, 3.5]
energy = np.logspace(2, 7.66, num=1000, endpoint=False)
fig, ax = plt.subplots()
for index in test_indices:
    ax.plot(energy, energy_likelihood(energy, index, np.deg2rad(45)), label=f"{index:.1f}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("E_reco [GeV]")
ax.set_ylabel("Energy likelihood")
ax.legend()
```

## Point source likelihood


Now we can bring together the spatial and energy terms to build a full `PointSourceLikelihood`. First, let's load some data from the simulation notebook to allow us to demonstrate.

```python
data = {}
with h5py.File("data/p_IC86_I_test_sim.h5", "r") as f:
    for key in f:
        if "source_0" not in key and "source_1" not in key:
            data[key] = f[key][()]
```

Now lets put our likelihood structure and data in together, along with a proposed source location:

```python
likelihood = PointSourceLikelihood(spatial_likelihood, energy_likelihood, 
                                  data["ra"], data["dec"], data["reco_energy"],
                                  source_coord)
likelihood._bg_index = 3.7
```

The likelihood will automatically select a declination band around the proposed source location. Because of the Gaussian spatial likelihood, neutrinos far from the source will have negligible contribution. We can control the width of this band with the optional argument `band_width_factor`. Let's see how many events ended up in the band, compared to the total number:

```python
likelihood.Nprime
```

```python
likelihood.Ntot
```

```python
energy_likelihood._min_index = 1.4
energy_likelihood._max_index = 4.0
likelihood._bg_energy_likelihood = None
```

We also note that the background likelihood is implemented automatically, for more information on the options here, check out the API docs. This is just a function of energy, with a constant factor to account for the isotropic directional likelihood.

```python
fig, ax = plt.subplots()
energy = np.logspace(new_reco_bins[0], new_reco_bins[-1], num=1000, endpoint=False)
ax.step(energy, energy_likelihood(energy, 3.7, np.deg2rad(30)), 
        label=f"index 3.7")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("$log_{10}(E_\mathrm{reco} / \mathrm{GeV})}$")
ax.set_ylabel("Background likelihood")
```

## Point source search

A point source search is usually carried out by defining the likelihood ratio of source and background hypotheses, then maximising this ratio as a function of $n_s$ and $\gamma$. The log likelihood ratio evaluated at the maximum likelihood values is then reffered to as the *test statistic*.

`icecube_tools` includes calculation of the test statistic, with optimisation performed by `iminuit`.

```python
np.log10(data["reco_energy"]).max()
```

```python
likelihood.get_test_statistic()
```

```python
likelihood._best_fit_index, likelihood._best_fit_ns
```

To understand the significance of this results, we would have to calculate the test statistic for a large number of background-only simulations. These could then be used to calculate a p-value. Given there is a strong point source in the simulation we used, we can expect the test stastic to be lower if we remove the source events. Let's try this:

```python
# Get all point source events
ps_sel = data["source_label"] == 1
ntot_ps_events = len(np.where(ps_sel==True)[0])

# Remove them one by one and find test statistic
test_statistics = []
for n_rm in range(ntot_ps_events):
    
    i_keep = (ntot_ps_events - n_rm) - 1
    ps_sel[np.where(ps_sel == True)[0][i_keep]] = False
    
    new_data = {}
    for key, value in data.items():
        new_data[key] = data[key][~ps_sel]
        
    new_likelihood = PointSourceLikelihood(spatial_likelihood, energy_likelihood,
                                       new_data["ra"], new_data["dec"], 
                                       new_data["reco_energy"],
                                       source_coord)
    new_likelihood._bg_energy_likelihood = None
    test_statistics.append(new_likelihood.get_test_statistic())
```

```python
likelihood._energy_likelihood._min_index
```

```python
fig, ax = plt.subplots()
ax.plot([_ for _ in range(ntot_ps_events)], test_statistics)
ax.set_xlabel("Number of point source events in dataset")
ax.set_ylabel("Test statistic value")
```

So the more neutrinos are seen from a source, the easier that source is to detect.

Let's have a look at the minuit object returned by `_minimize()`. In case something is off with the fit, we can use it as a starting point for debugging.

```python
m = likelihood._minimize()
print(m)
```

We can also draw the likelihood profiles for `ns` and `index` and compare to the inputted values (20 source events and an index of 2.5, cf. the simulation notebook)

```python
_ = m.draw_profile("ns")
```

```python
np.nonzero(data["source_label"] == 1)[0].size
```

```python
_ = m.draw_profile("index", bound=(likelihood._energy_likelihood._min_index, likelihood._energy_likelihood._max_index))
```

And finally have a look at the 2d likelihood profile.

```python
index = np.arange(1.7, 3.4, step=0.2)
index_pl = np.arange(1.6, 3.5, step=0.2)
ns_pl = np.arange(24., 46, step=2.)
ns = np.arange(25, 45, step=2.)
ii, nn = np.meshgrid(index, ns, indexing='ij')
ll = np.zeros(ii.flatten().shape)

for c, (i, n) in enumerate(zip(ii.flatten(), nn.flatten())):
    ll[c] = likelihood._func_to_minimize(n, i)

fig, ax = plt.subplots(dpi=150)

pcol = ax.pcolor(index_pl, ns_pl, ll.reshape(ii.shape).T)
pcol_ax = fig.colorbar(pcol)
pcol_ax.set_label("negative loglike")
ax.set_xlabel("index")
ax.set_ylabel("ns")
```

For the simulation-based energy likelihood used in the following, the profile o `index` has some edges. This is due to the interpolation done by the energy likelihood. We fed it simulation for discreet values of `index`. We cannot sensibly interpolate the likelihood of a given energy to any index, this leads to Heaviside-like discontinuities in the global (spatial + energy) likelihood. Instead, `PointSourceLikelihood` calculates the likelihood for the next simulated values of `index`, e.g. asking for an index 2.3 would lead to 2.0 and 2.5, and interpolate linearly between the resulting likelihoods. The precision of this method can be improved by providing simulations on a denser grid of spectral indices.

The error provided by `migrad()` is unreasonably small.

```python
energy_likelihood = MarginalisedEnergyLikelihood2021(np.round(np.arange(1.5, 4.1, 0.2), decimals=1), 'data', 'p_IC86_I', np.deg2rad(30))
energy_likelihood._min_index = 1.55
energy_likelihood._max_index = 3.85
likelihood = PointSourceLikelihood(spatial_likelihood, energy_likelihood, 
                                  data["ra"], data["dec"], data["reco_energy"],
                                  source_coord)
likelihood._bg_index = 3.7
likelihood._bg_energy_likelihood = None



```

```python
likelihood.get_test_statistic()
```

```python
m = likelihood._minimize()
```

```python
_ = m.draw_profile("ns")
```

```python
#Do this because for this interpolated function minuit can't compute reliable errors
index, llh =  m.draw_profile("index", bound=(likelihood._energy_likelihood._min_index, likelihood._energy_likelihood._max_index))
lower_lim = np.interp(llh.min() + 0.5, np.flip(llh[:np.nonzero(llh == llh.min())[0][0]]), np.flip(index[:np.nonzero(llh == llh.min())[0][0]]))
upper_lim = np.interp(llh.min() + 0.5, llh[np.nonzero(llh == llh.min())[0][0]:], index[np.nonzero(llh == llh.min())[0][0]:])
lims = plt.ylim()
plt.fill_betweenx([lims[0]-1, lims[1]+1], lower_lim, upper_lim, alpha=0.4, color='grey')
plt.ylim(lims)
plt.title(f"index = {m.values['index']:.1f} - {m.values['index']-lower_lim:.1f} + {abs(m.values['index']-upper_lim):.1f}")
```

# Time dependent point source analysis

```python
source_coords = (np.pi, np.deg2rad(30))
#index_list = list(np.arange(1.5, 4.25, 0.25))
event_files = ["data/p_IC86_I_test_sim.h5", "data/p_IC86_II_test_sim.h5"]
tllh = TimeDependentPointSourceLikelihood(
    source_coords, ["IC86_I", "IC86_II"], event_files, MarginalisedIntegratedEnergyLikelihood, path="data")

m = tllh._minimize()

m
```

```python
_ = m.draw_profile("index")
```

```python

```
