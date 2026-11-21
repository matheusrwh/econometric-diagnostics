"""Generate a synthetic country-year panel for trying the skill without private data.

Data-generating process (true slope on x = 0.5):
  y_it = a_i + d_t + 0.5 * x_it + u_it
  x_it correlated with a_i (so random effects is inconsistent),
  u_it AR(1) with rho = 0.4 and entity-specific variance, a few large outliers.

Usage:  python make_example_data.py   -> writes panel_example.csv next to this file
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N, T = 50, 40
rows = []
a = rng.normal(0, 1, N)
d = rng.normal(0, 0.5, T)
sigma = rng.uniform(0.5, 2.0, N)
for i in range(N):
    u_prev = 0.0
    for t in range(T):
        x = 0.8 * a[i] + rng.normal(0, 1)
        u = 0.4 * u_prev + rng.normal(0, sigma[i])
        u_prev = u
        y = a[i] + d[t] + 0.5 * x + u
        if rng.random() < 0.005:
            y += rng.choice([-1, 1]) * 15
        rows.append({"country": f"C{i:02d}", "year": 1980 + t, "x": x, "y": y})

out = Path(__file__).with_name("panel_example.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print(f"-> {out}")
