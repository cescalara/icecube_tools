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
    SpatialGaussianLikelihood
)
from icecube_tools.point_source_likelihood.energy_likelihood import (
    MarginalisedEnergyLikelihoodBraun2008, read_input_from_file,
)
from icecube_tools.point_source_likelihood.point_source_likelihood import (
    PointSourceLikelihood
)
```

## Spatial likelihood


We can start with the spatial/directional term. Let's approximate our detector as having a fixed angular resolution of 1 degree. We can then define the source spatial term as a 2D Gaussian with a fixed width. The background case will simply be an isotropic distribution on the sphere.

```python
angular_resolution = 1 # deg
spatial_likelihood = SpatialGaussianLikelihood(angular_resolution)
```

```python
source_coord = (np.pi, np.pi/4)
test_coords = [(np.pi+_, source_coord[1]) for _ in np.linspace(-0.1, 0.1, 100)]

fig, ax = plt.subplots()
ax.plot(np.rad2deg([tc[0] for tc in test_coords]), 
        [spatial_likelihood(tc, source_coord) for tc in test_coords])
ax.axvline(np.rad2deg(source_coord[0]), color="k", linestyle=":", 
           label="Source location")
ax.set_xlabel("RA [deg]")
ax.set_ylabel("Spatial likelihood")
ax.legend();
```

## Energy likelihood


Now let's think about the energy-dependent term. The way this is handled is to marginalise over the true neutrino energies, to directly connect the reconstructed neutrino energies to the spectral index of a simple power-law source model. 

Doing this properly requires a knowledge of the relationship between the true and reconstructed energies as well as the details of the power law model. The most straightforward way to implement this is to simulate the a large number of events using the `Simulator` and build a likelihood using the output of this simulation and `MarginalisedEnergyLikelihoodFromSim`. However, this can take a while to run, so here let's just use an implementation based on a plot from the original Braun et al. paper.

```python
energy_list, pdf_list, index_list = read_input_from_file("data/Braun2008Fig4b.h5")
energy_likelihood = MarginalisedEnergyLikelihoodBraun2008(energy_list, 
                                                          pdf_list, index_list)
```

```python
test_energies = np.geomspace(10, 1e7) # GeV
test_indices = [2, 2.5, 3, 4]

fig, ax = plt.subplots()
for index in test_indices:
    ax.plot(test_energies, [energy_likelihood(e, index) for e in test_energies], 
            label="index=%i" % index)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("E_reco [GeV]")
ax.set_ylabel("Energy likelihood")
ax.legend();
```

## Point source likelihood


Now we can bring together the spatial and energy terms to build a full `PointSourceLikelihood`. First, let's load some data from the simulation notebook to allow us to demonstrate.

```python
data = {}
with h5py.File("data/sim_output.h5", "r") as f:
    for key in f:
        if "source_0" not in key and "source_1" not in key:
            data[key] = f[key][()]
```

Now lets put our likelihood structure and data in together, along with a proposed source location:

```python
likelihood = PointSourceLikelihood(spatial_likelihood, energy_likelihood, 
                                  data["ra"], data["dec"], data["reco_energy"],
                                  source_coord)
```

The likelihood will automatically select a declination band around the proposed source location. Because of the Gaussian spatial likelihood, neutrinos far from the source will have negligible contribution. We can control the width of this band with the optional argument `band_width_factor`. Let's see how many events ended up in the band, compared to the total number:

```python
likelihood.N
```

```python
likelihood.Ntot
```

We also note that the background likelihood is implemented automatically, for more information on the options here, check out the API docs. This is just a function of energy, with a constant factor to account for the isotropic directional likelihood.

```python
fig, ax = plt.subplots()
ax.plot(test_energies, [likelihood._background_likelihood(e) for e in test_energies])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("E_reco [GeV]")
ax.set_ylabel("Background likelihood")
```

## Point source search

A point source search is usually carried out by defining the likelihood ratio of source and background hypotheses, then maximising this ratio as a function of $n_s$ and $\gamma$. The log likelihood ratio evaluated at the maximum likelihood values is then reffered to as the *test statistic*.

`icecube_tools` includes calculation of the test statistic, with optimisation performed by `iminuit`.

```python
likelihood.get_test_statistic()
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
    test_statistics.append(new_likelihood.get_test_statistic())
```

```python
fig, ax = plt.subplots()
ax.plot([_ for _ in range(ntot_ps_events)], test_statistics)
ax.set_xlabel("Number of point source events in dataset")
ax.set_ylabel("Test statistic value")
```

So the more neutrinos are seen from a source, the easier that source is to detect.

```python

```
