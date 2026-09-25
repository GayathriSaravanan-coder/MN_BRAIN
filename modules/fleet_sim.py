"""
fleet_sim.py
------------
OWNER: Member 3 (Telemetry Engine & Cloud/Environment Architect)

STUB implementation of live dumper-fleet telemetry, centered on the
Balaghat / Dongri Buzurg mining belt so app.py's map renders somewhere
real. Member 3 should replace the random-walk generator with the actual
GPS/telemetry feed (or a proper discrete-event simulation), keeping the
function name and returned columns identical.
"""

import numpy as np
import pandas as pd

# Approx coordinates: Balaghat / Dongri Buzurg manganese belt, MP, India
BASE_LAT, BASE_LON = 21.8000, 80.1830

STATUSES = ["Active", "Idle", "Maintenance", "Rerouted"]
STATUS_WEIGHTS = [0.70, 0.15, 0.08, 0.07]


def get_live_fleet_telemetry(num_dumpers: int = 40, shortfall_mode: bool = False, seed: int | None = None) -> pd.DataFrame:
    """
    Args:
        num_dumpers: fleet size to simulate.
        shortfall_mode: when True, biases more trucks into 'Rerouted' /
                        'Maintenance' to mimic a detected shortfall event.
        seed: optional RNG seed for reproducible demo runs.

    Returns:
        DataFrame[Vehicle_ID, Lat, Lon, Speed_kmh, Payload_Tons, Status, Sector]
    """
    rng = np.random.default_rng(seed)

    lat_jitter = rng.normal(0, 0.03, num_dumpers)
    lon_jitter = rng.normal(0, 0.03, num_dumpers)

    weights = STATUS_WEIGHTS
    if shortfall_mode:
        weights = [0.45, 0.15, 0.15, 0.25]

    status = rng.choice(STATUSES, size=num_dumpers, p=weights)
    speed = np.where(status == "Active", rng.uniform(15, 42, num_dumpers), 0.0)
    payload = np.where(
        np.isin(status, ["Active", "Rerouted"]),
        rng.uniform(18, 35, num_dumpers),
        0.0,
    )
    sector = rng.integers(1, 6, num_dumpers)

    df = pd.DataFrame(
        {
            "Vehicle_ID": [f"DMP-{i + 1:03d}" for i in range(num_dumpers)],
            "Lat": BASE_LAT + lat_jitter,
            "Lon": BASE_LON + lon_jitter,
            "Speed_kmh": speed.round(1),
            "Payload_Tons": payload.round(1),
            "Status": status,
            "Sector": sector,
        }
    )
    return df


def compute_shift_kpis(fleet_df: pd.DataFrame, target_tonnage: float = 900.0) -> dict:
    """Rolls the fleet snapshot up into the Zone-1 KPI numbers for Tab 3."""
    active = int((fleet_df["Status"] == "Active").sum())
    produced = float(fleet_df["Payload_Tons"].sum())
    rerouted = int((fleet_df["Status"] == "Rerouted").sum())
    risk = "Alert" if produced < target_tonnage * 0.85 else "Normal"
    return {
        "active_dumpers": active,
        "shift_production_t": round(produced, 1),
        "target_tonnage": target_tonnage,
        "rerouted": rerouted,
        "risk": risk,
    }