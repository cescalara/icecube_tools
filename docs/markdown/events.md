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

```python
from icecube_tools.utils.data import SimEvents, RealEvents

import numpy as np
import matplotlib.pyplot as plt
```

<!-- #region -->
## Event classes


Events of simulations and events detected by icecube are stored in instances of `SimEvents` and `RealEvents`, respectively. We load some simulated events:
<!-- #endregion -->

```python
events = SimEvents.load_from_h5("multi_test.hdf5")
```

Events can span multiple data seasons. For each season, one entry is stored in `periods`.

```python
events.periods
```

The properties of the event are stored as properties. A list of all data fields is accessible through

```python
events.keys
```

The entries differ for simulated and actual events.

For each period a dictionary of all such data fields is returned by `period(p)`.

```python
events.period("IC86_II")
```

Through the method `apply_mask()` a mask is applied and a nested dictionary (1st key is period, 2nd key is name of the data field, as returned by `keys`) is returned containing the selected events. The order of entries in mask needs to be the same as the periods in `events.periods`. For application, see the notebook on point source analysis.
