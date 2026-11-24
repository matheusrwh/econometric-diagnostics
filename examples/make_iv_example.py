"""Generate synthetic data for the IV and shift-share examples (no private data needed).

1. iv_example.csv — cross section of 3,000 individuals in 60 states.
   x = 0.5 z1 + 0.07 z2 + 0.5 w + v,   y = 1 + 1.0 x + 0.5 w + u,   u = 0.6 v + e
   (true effect of x = 1.0; OLS is biased upwards; z2 is a weak instrument).

2. Shift-share (Bartik) example — 600 regions in 40 states, 80 sectors in 20 groups.
   shares s_ik (incomplete: they sum to the region's tradable share, 0.3-0.9),
   shocks g_k ~ N(0, 1), sector-level error components eta_k ~ N(0, 1):
     eps_i = 2 * sum_k s_ik eta_k + e_i         (regions with similar shares share errors)
     x_i   = 1.0 * sum_k s_ik g_k + 0.5 eps_i + v_i
     y_i   = 0.8 x_i + eps_i                    (true effect 0.8; instrument valid)
   Files: ss_regions.csv (with the instrument z_bartik), ss_exposure.csv (long shares),
          ss_shocks.csv.
   With ~17 effective shocks a single draw can land far from 0.8 (this one gives ~1.6):
   region-level SEs and the AR set miss it, the AKM0 set covers it — the point of AKM.

Usage:  python make_iv_example.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

here = Path(__file__).parent
rng = np.random.default_rng(7)

# ---- 1. cross-section IV
n, G = 3000, 60
state = rng.integers(0, G, n)
w = rng.normal(0, 1, n)
z1 = rng.normal(0, 1, n)
z2 = rng.normal(0, 1, n)
v = rng.normal(0, 1, n)
e = rng.normal(0, 1, n) * (0.5 + 0.5 * np.abs(w)) + rng.normal(0, 0.5, G)[state]
x = 0.5 * z1 + 0.07 * z2 + 0.5 * w + v
y = 1 + 1.0 * x + 0.5 * w + 0.6 * v + e
pd.DataFrame({"id": np.arange(n), "state": [f"S{s:02d}" for s in state],
              "y": y, "x": x, "w": w, "z1": z1, "z2": z2}).to_csv(here / "iv_example.csv", index=False)

# ---- 2. shift-share
N, K = 600, 80
region_state = rng.integers(0, 40, N)
alpha = rng.gamma(0.3, 1.0, K)                     # a few large sectors
raw = rng.dirichlet(alpha + 0.02, N)
tradable = rng.uniform(0.3, 0.9, N)
S = raw * tradable[:, None]
S = np.where(S > 1e-4, S, 0.0)                      # tiny shares are not stored in the file
g = rng.normal(0, 1, K)
eta = rng.normal(0, 1, K)
eps = 2 * S @ eta + rng.normal(0, 0.5, N)
z = S @ g
xr = 1.0 * z + 0.5 * eps + rng.normal(0, 0.5, N)
yr = 0.8 * xr + eps
pd.DataFrame({"region": [f"R{i:03d}" for i in range(N)],
              "state": [f"S{s:02d}" for s in region_state],
              "pop": rng.integers(10, 500, N), "y": yr, "x": xr, "z_bartik": z,
              "tradable_share": tradable}).to_csv(here / "ss_regions.csv", index=False)
ii, kk = np.nonzero(S > 1e-4)
pd.DataFrame({"region": [f"R{i:03d}" for i in ii], "sector": [f"K{k:02d}" for k in kk],
              "share": S[ii, kk]}).to_csv(here / "ss_exposure.csv", index=False)
pd.DataFrame({"sector": [f"K{k:02d}" for k in range(K)], "g": g,
              "sector_group": [f"G{k // 4:02d}" for k in range(K)]}).to_csv(
    here / "ss_shocks.csv", index=False)
print("-> iv_example.csv, ss_regions.csv, ss_exposure.csv, ss_shocks.csv")
