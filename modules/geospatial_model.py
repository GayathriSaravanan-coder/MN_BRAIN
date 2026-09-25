"""
geospatial_model.py
--------------------
OWNER: Member 2 (Geospatial AI & Optimization Engine Developer)

This is a STUB / mock implementation so that Member 1's dashboard (app.py)
can be built, wired, and demoed *before* the real Kriging engine is ready.

Real deliverable (per team plan):
    get_kriged_block_model(pit_name, depth_m, confidence_cutoff)
        -> pandas.DataFrame[X, Y, Z, Mn_Grade, Fe_Grade, UNFC_Code]
    using Ordinary Kriging with a spherical variogram over borehole logs.

Member 2 should replace the body of this function with the real
geostatistics (e.g. using pykrige / gstools), keeping the exact same
function name, arguments, and returned column names so app.py does not
need to change.
"""

import numpy as np
import pandas as pd


def _spherical_variogram(h, nugget=0.05, sill=1.0, range_=40.0):
    """Simple spherical variogram, used only to shape the mock noise."""
    h = np.asarray(h, dtype=float)
    gamma = np.where(
        h <= range_,
        nugget + (sill - nugget) * (1.5 * (h / range_) - 0.5 * (h / range_) ** 3),
        sill,
    )
    return gamma


def get_kriged_block_model(pit_name: str, depth_m: float, confidence_cutoff: float) -> pd.DataFrame:
    """
    Mock 3D block model generator.

    Args:
        pit_name: name of the selected mining pit/site.
        depth_m: max depth (m) to model down to.
        confidence_cutoff: 0-100, blocks below this prediction confidence
                            are dropped (mimics real kriging variance filtering).

    Returns:
        DataFrame[X, Y, Z, Mn_Grade, Fe_Grade, UNFC_Code, Confidence]
    """
    rng = np.random.default_rng(abs(hash(pit_name)) % (2**32))

    block_size = 5  # 5m x 5m x 5m blocks, per the proposed solution
    x_extent, y_extent = 150, 150
    xs = np.arange(0, x_extent, block_size)
    ys = np.arange(0, y_extent, block_size)
    zs = np.arange(0, depth_m, block_size)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    X, Y, Z = X.ravel(), Y.ravel(), Z.ravel()

    # Distance from an arbitrary "rich vein" center, used to fake spatial
    # correlation the way a real variogram-based kriging surface would.
    cx, cy, cz = x_extent * 0.4, y_extent * 0.6, depth_m * 0.5
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)

    variance = _spherical_variogram(dist, range_=depth_m + 20)
    confidence = np.clip(100 * (1 - variance), 40, 99)

    base_mn = 46 - 0.06 * dist + rng.normal(0, 1.5, size=dist.shape)
    mn_grade = np.clip(base_mn, 8, 54)

    fe_grade = np.clip(18 + 0.03 * dist + rng.normal(0, 1.0, size=dist.shape), 4, 30)

    unfc_code = np.where(
        (mn_grade >= 30) & (confidence >= 80),
        "111",
        np.where(mn_grade >= 20, "122", "333"),
    )

    df = pd.DataFrame(
        {
            "X": X,
            "Y": Y,
            "Z": -Z,  # negative so depth renders downward in 3D plots
            "Mn_Grade": mn_grade.round(2),
            "Fe_Grade": fe_grade.round(2),
            "UNFC_Code": unfc_code,
            "Confidence": confidence.round(1),
        }
    )

    return df[df["Confidence"] >= confidence_cutoff].reset_index(drop=True)