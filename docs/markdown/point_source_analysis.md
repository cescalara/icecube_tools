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
from icecube_tools.utils.data import RealEvents, SimEvents
from icecube_tools.point_source_analysis.point_source_analysis import MapScan
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
```

```python
events = RealEvents.from_event_files("IC86_I", "IC86_II")
```

```python
events.ra
```

```python
scan = MapScan(
    "example.yaml",
    events
)
```

```python
scan.generate_sources()
```

```python
scan.perform_scan()
```

```python
l = ["n0", "n1", "index"]

d = {"n0": 1, "n1":2, "index":3}

a = [d[i] for i in l if i != "index"]
print(a)
```

```python
for i in d.values():
    print(i)
```

```python
scan.write_config("written_config.yaml")
```

```python
scan.periods
```

```python

```
