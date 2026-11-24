# Replication package – 3D ShapeFinder forecasts of conflict fatalities (PRIO-Grid)

This package reproduces every table and figure in the paper: the matching, the
parameter validation, the forecasts and their evaluation against VIEWS and the
two autoregressive benchmarks.

## Structure

```
Scripts/            Python scripts, run in the numbered order
Data/               Input data
Data/Results/       Intermediate results (matches, zones, forecasts, grid search)
Figure/             Figures, named after their number in the paper
requirements.txt    Python version and package versions
```

## Installation

Python 3.8.5 with the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

The scripts can be run from any folder: each one sets its working directory
to `Scripts/` and reads/writes `../Data/` and `../Figure/`.

## Data

| File | Content |
|---|---|
| `Data/df_input_pg.csv` | Monthly fatalities (UCDP GED) per PRIO-Grid cell, January 1990 – December 2023 (rows: VIEWS `month_id`, columns: PRIO-Grid ids) |
| `Data/prio.csv` | PRIO-Grid cells with their row/column position and coordinates |
| `Data/views_pred_1pg.csv` | VIEWS forecast issued in January 2022 (test period Jan–Jun 2022) |
| `Data/views_pred_06_2022.csv` | VIEWS forecast issued in July 2022 (test period Jul–Dec 2022) |
| `Data/views_pred_2pg.csv` | VIEWS forecast issued in January 2023 (test period Jan–Jun 2023) |
| `Data/views_pred_06_2023.csv` | VIEWS forecast issued in July 2023 (test period Jul–Dec 2023) |

## Test periods and file suffixes

| Forecast period | Input window (12 months) | Suffix in `Data/Results/` |
|---|---|---|
| January – June 2022 | Jan – Dec 2021 | `_2022` |
| July – December 2022 | Jul 2021 – Jun 2022 | `_sub2` |
| January – June 2023 | Jan – Dec 2022 | *(none)* |
| July – December 2023 | Jul 2022 – Jun 2023 | `_sub1` |
| Validation: January – June 2021 | Jan – Dec 2020 | `_val` |

## Scripts

All intermediate results are already in `Data/Results/`. To reproduce only
the tables and figures, run `3_Main_Analysis.py` onwards; steps 0–2 are
needed only to recompute everything from scratch.

| Script | What it does | Reads | Writes | Run time |
|---|---|---|---|---|
| `0_Download_VIEWS.py` | Downloads the four VIEWS PRIO-Grid forecasts from the VIEWS API | VIEWS API (internet) | `Data/views_pred_*.csv` | minutes |
| `1_Matching.py` | Detects the active zones of each test period and searches the history for similar 3D sequences (EMD + ratio of active cells R) | `df_input_pg.csv`, `prio.csv` | `Results/final_traj*.pkl`, `Results/all_final_traj.pkl`, `Results/input*.pkl`, `Results/matches*.pkl` | ~1 h per period (~4 h total) |
| `2_Validation.py` | Matching on the validation period, then a grid search over the EMD threshold, R threshold, cluster distance and minimum number of matches; selects the lowest PRIO-Grid MSE | `df_input_pg.csv`, `prio.csv` | `Results/*_val.pkl`, `Results/df_grid_val.csv` | long (matching + grid search) |
| `3_Main_Analysis.py` | Builds the forecasts with the selected parameters and computes Table 1, Tables E.1–E.3, the numbers quoted in the text, and the figures listed below | `Data/*.csv`, `Data/Results/*` | `Figure/*`, `Results/3d_sf.csv` | – |
| `4_Toy_Model.py` | Toy example of three patterns on a 5×5 grid (Figure 3 and the calculations in Appendix J) | – | `Figure_3.png`, `Figure_J1.png` | seconds |
| `5_Appendix_H.py` | Rolling simulation of the active-zone selection (share of fatalities and active cells covered) | `df_input_pg.csv`, `prio.csv` | `Figure_H1.png` | minutes |
| `6_Appendix_I.py` | Comparison with the original (2D) ShapeFinder over July–December 2023 (Table I.4 and paired t-tests, printed) | `df_input_pg.csv`, `prio.csv`, `Results/3d_sf.csv`, `Results/2d_sf.csv`, `Results/final_traj_sub1.pkl` | printed table | seconds (~1 day with `run_2d_sf = True`) |
| `shape.py` | Original (2D) ShapeFinder, imported by `6_Appendix_I.py` | – | – | – |

`6_Appendix_I.py` reads the stored 2D forecasts (`Results/2d_sf.csv`). To
recompute them, set `run_2d_sf = True` at the top of the script (about one
day of computation).

## Figures produced by `3_Main_Analysis.py`

| Paper | File(s) |
|---|---|
| Figure 1 | `Figure_1a.png` – `Figure_1d.png` |
| Figure 2 | `Figure_2_input.*`, `Figure_2_similar_*.*`, `Figure_2_map_*.png` (3D views as interactive `.html` and `.glb` meshes) |
| Figure 4 | `Figure_4a.png` – `Figure_4d.png` |
| Figure 8 | `Figure_8_scenario.html` |
| Figure 9 | `Figure_9_input.html`, `Figure_9_past_future_*.html`, `Figure_9_forecast.html`, `Figure_9_observed.html` |
| Figures 10–13 | `Figure_10.png` – `Figure_13.png` |
| Appendix A–D | `Figure_A1_top.png`, `Figure_A1_bottom.png`, `Figure_B1.png`, `Figure_B2.png`, `Figure_C1.png`, `Figure_D1.png` |
| Appendix F | `Figure_F1.png`, `Figure_F2a.png` – `Figure_F2d.png` |
| Appendix H | `Figure_H2.png`, `Figure_H3.png` (Figure H.1 comes from `5_Appendix_H.py`) |

Figures 5–7 are diagrams of the method and are not produced by code.

## Selected parameters

The grid search in `2_Validation.py` (lowest MSE at the PRIO-Grid level on
the validation set) selects: EMD threshold 0.15, R threshold 0.09, cluster
distance Ndim/1100, and at least 11 matches per zone.
