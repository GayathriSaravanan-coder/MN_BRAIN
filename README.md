# MnBrain

### MOIL Manganese Operations Engine

Subsurface modeling · Grade blending optimization · Fleet logistics · AI operations co-pilot

---

Built for **Smart India Hackathon 2026**, Problem Statement **26009** (Ministry of Steel, for MOIL Ltd.).

## The Problem

MOIL Ltd. operates multiple manganese mines across India. Today, grade estimation, ore blending, and fleet dispatch run as three disconnected processes — planners rarely have subsurface grade, blend economics, and live fleet capacity in one place at decision time. MnBrain consolidates all three into a single operations dashboard, with ore reserve reporting aligned to **UNFC Code 111**.

## Modules

### 1. Subsurface Modeling
- 3D Kriging interpolation of Mn% grade across a mine site (e.g. Balaghat Pit-3), with site, depth, and confidence-cutoff controls that re-run the model live
- Headline KPIs: total ore volume (m³), average Mn grade (%), and % of the block model that is UNFC Code 111 verified
- Block model cross-checked against GSI lithology maps and **IBM MCDR Rule 34A**
- **Human-in-the-loop recalibration:** geologists log physical core sample results — borehole ID, lab-verified Mn grade, ground verification status (e.g. Confirmed Match), and lithology/alteration notes — back into the app, and the Kriging surface recalculates against the new ground truth

### 2. Grade Blending
- MILP solver over stockpile inventory (Mn%, Fe%, available tonnage) across grade bands — SP-A High Grade, SP-B Mid Grade, SP-C Low Grade, SP-D Reject Fines
- Inputs: target Mn grade and target tonnage; solved on demand via **Run Prescriptive Blending**
- Outputs: per-stockpile blend allocation (bar chart), optimized Mn grade, and projected cost savings in ₹
- **Explainable decisions** — the solver reports its own reasoning after each run, e.g.:
  - *High-Grade Primary Allocation* — draws heavily from the high-grade stockpile to lift the batch above target
  - *Impurity Ceiling Guard* — caps a high-Fe stockpile's share to stay within the IBM MCDR Fe threshold
  - *Economic Yield Balance* — maximizes the mid-grade stockpile's use to cut per-tonne cost without sacrificing revenue
  - A convergence line reports solver time and the final Mn% / Fe% achieved against target

### 3. Fleet & Logistics
- Live telemetry across active dumpers and mine sectors, with configurable simulation speed and a **Simulate Sector Breakdowns** stress-test toggle
- KPIs: active dumpers, shift production against target tonnage, and a shortfall risk flag per sector (Normal / Alert)
- On a detected shortfall, a discrete-event reroute plan is triggered automatically — idle dumpers from unaffected sectors are reassigned and the recovered tonnage is reported back on screen
- **Backend System Health & Data Ingestion Logs** panel streams real-time system events, e.g. telemetry gateway connection over MQTT (mTLS secured), ingested GPS/IoT packet counts, shortfall detection, the reroute action taken, and system memory/latency stats

### MnBrain Co-Pilot
A conversational assistant docked in the sidebar, available from any tab. It answers natural-language operations questions — reserve status, blending logic, fleet alerts — grounded in the dashboard's live state. Asking for the current shortfall risk, for example, returns which sector is affected and the specific reroute action available to fix it.

## Tech Stack

| Layer | Technology |
|-------|------------|
| Dashboard / UI | Streamlit |
| Visualization | Plotly (3D scatter, bar charts) |
| Geospatial modeling | Kriging interpolation |
| Blend optimization | MILP solver with post-hoc explainability |
| Fleet simulation | Discrete-event simulation + MQTT telemetry ingestion |
| Operations assistant | Conversational co-pilot grounded in live app state |

## Project Structure

```
MN_BRAIN/
├── app.py                     # Streamlit entry point — page config, theming, tab routing
├── modules/
│   ├── geospatial_model.py    # Kriging block model (get_kriged_block_model)
│   ├── optimization_engine.py # MILP blend solver (calculate_optimal_blend, get_default_stockpiles)
│   └── fleet_sim.py           # Telemetry + shift KPIs (get_live_fleet_telemetry, compute_shift_kpis)
├── assets/
│   └── logo.png
├── requirements.txt
└── README.md
```

## Getting Started

**Prerequisites:** Python 3.10+

```bash
# Clone the repo
git clone https://github.com/GayathriSaravanan-coder/MN_BRAIN.git
cd MN_BRAIN

# Set up a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
python -m streamlit run app.py
```

The dashboard runs at `http://localhost:8501`.

## Roadmap

- [ ] Persist field verification logs to a database instead of session state
- [ ] Extend the Co-Pilot with historical shortfall trend analysis
- [ ] Add multi-site comparison view in Subsurface Modeling
- [ ] Replace simulated MQTT telemetry with a live gateway integration

## Team

| | |
|---|---|
| **Gayathri S** | Frontend & System Integration Lead — dashboard architecture, Plotly visualizations, wiring UI to backend modules |
| **Mahalakshmi** | Geospatial Kriging model & MILP blending optimization engine |
| **Anbarasi** | Fleet telemetry simulation, MQTT ingestion, and deployment configuration |

## License

Built for Smart India Hackathon 2026. Not currently licensed for external use.
