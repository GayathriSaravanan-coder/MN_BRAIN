"""
optimization_engine.py
------------------------
OWNER: Member 2 (Geospatial AI & Optimization Engine Developer)

STUB implementation of the MILP grade-blending solver, wired to the exact
interface app.py expects. Member 2 should swap the body of
calculate_optimal_blend() for a real linprog()/pulp formulation matching
the constraints in the proposed solution (min Mn grade, max Fe impurity,
stockpile availability), but must keep the function name, args, and the
returned DataFrame/summary shape unchanged.
"""

import time

import numpy as np
import pandas as pd
from scipy.optimize import linprog


def get_default_stockpiles() -> pd.DataFrame:
    """Mock stockpile table -- Member 1 shows/edits this in the left panel."""
    return pd.DataFrame(
        {
            "Stockpile": ["SP-A (High Grade)", "SP-B (Mid Grade)", "SP-C (Low Grade)", "SP-D (Reject Fines)"],
            "Mn_Percent": [52.0, 44.0, 34.0, 22.0],
            "Fe_Percent": [6.0, 9.5, 14.0, 21.0],
            "Available_Tonnes": [2000, 4000, 3000, 1500],
            "Cost_per_Tonne": [3400, 2650, 1900, 1100],
        }
    )


def calculate_optimal_blend(stockpile_df: pd.DataFrame, target_tonnage: float, target_mn_percent: float,
                             fe_max_percent: float = 12.0):
    """
    Solve: minimize blending cost subject to
        sum(w_j) == target_tonnage
        sum(Mn_j * w_j) >= target_mn_percent * target_tonnage
        sum(Fe_j * w_j) <= fe_max_percent  * target_tonnage
        0 <= w_j <= available_j

    Returns:
        (result_df, summary_dict)
        result_df: stockpile_df + Allocated_Tonnes, Contribution_Percent
        summary_dict: {solved, blended_mn, blended_fe, total_cost, solve_time_s}
    """
    start = time.perf_counter()

    n = len(stockpile_df)
    c = stockpile_df["Cost_per_Tonne"].to_numpy()
    mn = stockpile_df["Mn_Percent"].to_numpy()
    fe = stockpile_df["Fe_Percent"].to_numpy()
    avail = stockpile_df["Available_Tonnes"].to_numpy()

    A_ub = np.vstack([-mn, fe])
    b_ub = np.array([-target_mn_percent * target_tonnage, fe_max_percent * target_tonnage])

    A_eq = np.ones((1, n))
    b_eq = np.array([target_tonnage])

    bounds = [(0, avail[i]) for i in range(n)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    solve_time = time.perf_counter() - start
    result_df = stockpile_df.copy()

    if res.success:
        allocated = res.x
    else:
        # Infeasible target -- fall back to a proportional best-effort blend
        # so the UI never breaks, and flag it as not solved.
        allocated = np.minimum(avail, target_tonnage / n)

    result_df["Allocated_Tonnes"] = allocated.round(1)
    result_df["Contribution_Percent"] = (allocated / allocated.sum() * 100).round(1) if allocated.sum() else 0

    blended_mn = float((allocated * mn).sum() / allocated.sum()) if allocated.sum() else 0
    blended_fe = float((allocated * fe).sum() / allocated.sum()) if allocated.sum() else 0
    total_cost = float((allocated * c).sum())

    summary = {
        "solved": bool(res.success),
        "blended_mn": round(blended_mn, 2),
        "blended_fe": round(blended_fe, 2),
        "total_cost": round(total_cost, 0),
        "solve_time_s": round(solve_time, 3),
    }
    return result_df, summary