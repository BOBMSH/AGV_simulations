# AGV Control Strategy Benchmark

Python simulation comparing four control strategies for an Automated Guided
Vehicle (AGV) used in manufacturing logistics:

1. **Traditional PID**
2. **Intelligent PID** (NumPy MLP gain scheduler)
3. **Traditional MPC** (cvxpy/OSQP linear MPC with constraints)
4. **Intelligent MPC** (NumPy MLP plant-residual model + Jacobian linearisation)

Four scenarios, each engineered so that one specific controller is the
structural winner:

| Scenario | Description | Winner | Metric |
|---|---|---|---|
| A | Smooth concrete warehouse aisle cruise | Traditional PID | lowest compute at competitive RMSE |
| B | 4 floor surfaces + 4 payload pickup stations | Intelligent PID | lowest RMSE under varying surface+payload |
| C | Tight dock with v_max(s) envelope and ±5 cm stop | Traditional MPC | parks within ±5 cm of the dock |
| D | Inter-plant: payload pickup, wet uphill, downhill | Intelligent MPC | lowest RMSE on the combined disturbance |

No PyTorch / TensorFlow dependency — the NNs are implemented in pure NumPy.

## Setup

```bash
pip install -r requirements.txt
# Optional: regenerate the NN checkpoints (already shipped):
python -m agv_sim.nn.training         # IPID gain scheduler
python -m agv_sim.nn.impc_training    # IMPC plant residual
```

## Quickstart

```bash
# Single controller:
python -m agv_sim --scenario A --controller pid
python -m agv_sim --scenario B --controller ipid
python -m agv_sim --scenario C --controller mpc
python -m agv_sim --scenario D --controller impc

# Multi-controller static comparison figure:
python -m agv_sim --scenario D --controllers pid,ipid,mpc,impc

# Cross-scenario sweep + summary heatmap:
python -m agv_sim.utils.sweep                                              # KPI sweep
python -c "from agv_sim.viz import render_summary; from agv_sim.utils.sweep import load_csv; from pathlib import Path; render_summary(load_csv(Path('results/kpi_sweep.csv')), Path('results/summary_heatmap'))"

# Live 2D animation (MP4 if ffmpeg available, else GIF):
python -m agv_sim.viz.live --scenario D --controller impc --save results/live_D_impc --speedup 2.5

# 4-way live comparison animation:
python -m agv_sim.viz.live_compare --scenario D --controllers pid,ipid,mpc,impc --save results/live_D_compare

# Smoke tests:
python -m tests.test_phase1_smoke
python -m tests.test_phase2_smoke
python -m tests.test_phase3_smoke
python -m tests.test_phase4_smoke
```

Outputs are written to `results/` as 300-dpi PNG, vector PDF, and MP4.

## Headline results

| Scenario | PID RMSE | IPID RMSE | MPC RMSE | IMPC RMSE | Winner |
|---|---:|---:|---:|---:|---|
| A | 0.0255 | 0.0255 | 0.0217 | 0.0182 | **PID** (1700x cheaper compute, indistinguishable tracking) |
| B | 0.0232 | **0.0166** | 0.0351 | 0.0181 | **IPID** (28% lower RMSE than PID) |
| C | 0.0272 | 0.0272 | 0.0726 | 0.0647 | **MPC** (parks within ±5cm; PID overshoots dock by 5.5cm) |
| D | 0.0606 | 0.0457 | 0.0701 | **0.0243** | **IMPC** (65% lower RMSE than MPC) |

(RMSE in m/s. Compute per tick: PID ~3 µs, IPID ~25 µs, MPC/IMPC ~4 ms.)

## Presentation-ready figure pack

`results/figure_pack/` contains the curated set of files for the talk:

- `00_summary_heatmap.{png,pdf}` — cross-scenario RMSE + compute heatmaps
- `01..04_scenario_{A,B,C,D}.{png,pdf}` — per-scenario static comparison figures
- `10..13_live_{A_pid, B_ipid, C_mpc, D_impc}.mp4` — single-controller live demos
- `14_live_D_4way_comparison.mp4` — all four controllers on Scenario D side-by-side
- `kpi_sweep.csv` — raw numbers for any custom table you want to build

## How each controller works

### Traditional PID
Parallel-form discrete PID with filtered derivative and back-calculation
anti-windup. Per-scenario default gains.

### Intelligent PID (IPID)
Same control loop, but Kp/Ki/Kd are re-computed every tick by a small MLP.

1. **Offline data generation** (`agv_sim/nn/training.py`): sample (Cr, payload)
   operating points; an analytic gain schedule derived from the linearised
   plant dynamics yields target (Kp, Ki, Kd) labels.
2. **MLP training**: a 2->32->32->3 fully-connected net with softplus
   output activation. NumPy implementation with manual backprop and Adam.
3. **Online inference** (`agv_sim/controllers/ipid.py`): the AGV's onboard
   estimate of (Cr, payload) is passed through the MLP each tick. Resulting
   gains feed into the standard PID step.

### Traditional MPC
Linear receding-horizon MPC over (s, v) state with split throttle/brake
controls so the QP correctly models the asymmetric plant force capability
(60 kN drive vs 80 kN brake) without non-convex complementarity constraints.
Soft slack variables on v_max with very high penalty keep the QP feasible
when entering a speed-limit zone faster than the brake can recover.
DPP-compliant -> OSQP reuses factorisation -> median solve ~5 ms.

### Intelligent MPC (IMPC)
Same QP framework but augmented with a learned plant residual:

1. **Residual dataset** (`agv_sim/nn/impc_training.py`): 3000 random
   (v, u_thr, u_brk, cr, payload, grade) configurations sampled from the
   true plant; the residual is the gap between actual velocity change
   and the MPC's nominal model prediction. The NN achieves **99.9%
   reduction in residual MSE** vs the zero-baseline.
2. **MLP architecture**: 6 -> 32 -> 32 -> 1, linear output (residuals can
   be negative).
3. **Online SQP-style linearisation** (`agv_sim/controllers/impc.py`):
   each tick the NN is queried at the current (v, u_thr_prev, u_brk_prev,
   cr, payload, grade), returning the residual value and its **analytical
   Jacobian** (manually derived through the ReLU layers — no autograd
   needed). The augmented dynamics constraint is parameterised and
   affine; the QP stays DPP-compliant, OSQP reuses its factorisation,
   median solve ~5 ms.

## Repo structure

```
agv_sim/
  plant/          AGV longitudinal dynamics (RK4 @ 100 Hz)
  controllers/    base.py, pid.py, mpc.py, ipid.py, impc.py
  scenarios/      base.py, scenario_a_simple.py, scenario_b_friction.py,
                  scenario_c_dock.py, scenario_d_combined.py
  viz/            static.py (single run), compare.py (multi controller),
                  heatmap.py (cross-scenario summary),
                  live.py (live 2D animation), live_compare.py (4-up)
  utils/          kpi.py, runner.py, sweep.py
  nn/             mlp.py (NumPy MLP + Jacobian),
                  training.py (IPID), impc_training.py (IMPC residual),
                  checkpoints/
results/          Generated figures + CSVs + MP4s
  figure_pack/    Curated subset ready for the .pptx
tests/            Phase 1-4 smoke tests
```
