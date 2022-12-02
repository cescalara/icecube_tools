---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
from icecube_tools.utils.data import RealEvents, SimEvents
from icecube_tools.point_source_analysis.point_source_analysis import MapScan
import numpy as np
import matplotlib.pyplot as plt
```

## MapScan

We can perform a scan over the sky for point sources. At each proposed source location a likelihood fit is performed. First we have to provide some events, here we use the simulated ones from the example notebook.

```python
events = SimEvents.load_from_h5("h5_test.hdf5")
```

Then we create a MapScan() object with some configuration `config.yaml` in which source lists, data cuts, etc. can be stored, the events, and a path for the output of the scan `test_output.hdf5`.

```python
scan = MapScan(
    "config.yaml",
    events,
    "test_output.hdf5"
)
```

Let's create a small grid around the source location of the simulation: (ra, dec) = (180°, 30°)

```python
dec = np.linspace(np.deg2rad(25), np.deg2rad(35), 11)
ra = np.linspace(np.pi - np.deg2rad(5), np.pi + np.deg2rad(5), 11)
rr, dd = np.meshgrid(ra, dec)
```

The source coordinates need to be handed over the the MapScan, then `generate_sources()` is called. Although the sources are already generated, it is able to create source lists from healpy keywords npix and nside for entire sky searches. Furthermore, output arrays of appropriate sizes are created. Afterwards the fits are started.

```python
scan.ra_test = rr.flatten()
scan.dec_test = dd.flatten()
scan.generate_sources()
scan.perform_scan(show_progress=True)
```

We can have a look at the results:

```python
fig, ax = plt.subplots(1, 3, figsize=(15, 4))

pcol = ax[0].pcolormesh(ra, dec, scan.ts.reshape((11, 11)), shading="nearest")
fig.colorbar(pcol, ax=ax[0], label="TS")

pcol = ax[1].pcolormesh(ra, dec, scan.index.reshape((11, 11)), shading="nearest")
fig.colorbar(pcol, ax=ax[1], label="index")

pcol = ax[2].pcolormesh(ra, dec, scan.ns.reshape((11, 11)), shading="nearest")
fig.colorbar(pcol, ax=ax[2], label="ns")
fig.savefig("example_sky_skan.png", dpi=150)
```

```python

```
