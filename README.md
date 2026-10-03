# VM Placement Experiment

This CSE-307 experiment compares static First-Fit, static Best-Fit, and adaptive
Predictive Placement under a synthetic workload shift.

The simulation uses 18 VMs, 6 equal-capacity hosts, 50 time steps, and random
seed 7. First-Fit and Best-Fit place the VMs once at time 0. Predictive Placement
starts from the Best-Fit placement and uses Linear Regression to predict the next
VM loads from recent history. If a host is predicted to be overloaded, the
largest feasible VM is moved to the least-loaded host with enough predicted
capacity. Training uses only observations available before the predicted step.

The workload becomes heavier at time 25. Mean VM demand is 13.459 before the
shift and 17.783 after it. All methods use this same workload trace.

## Metrics

- Utilization imbalance: standard deviation of the six host utilizations
- SLA violations: number of host-time observations above host capacity
- Migrations: number of VMs actually moved between hosts

## Run

```text
python -m pip install -r requirements.txt
python experiment.py
```

The script creates `results/summary.csv`, `results/time_series.csv`, and
`results/comparison.png`.

## Verified Results

| Method | Imbalance before | Imbalance after | SLA before | SLA after | SLA total | Migrations |
|---|---:|---:|---:|---:|---:|---:|
| First-Fit | 0.3253 | 0.4465 | 3 | 89 | 92 | 0 |
| Best-Fit | 0.3253 | 0.4465 | 3 | 89 | 92 | 0 |
| Predictive Placement | 0.3098 | 0.1379 | 3 | 53 | 56 | 4 |

First-Fit and Best-Fit use different placement rules but produce the same
initial assignment for this particular deterministic workload.

## AI Assistance

AI assistance was used to help implement, test, and document the experiment.
The student remains responsible for reviewing the method, results, and report.
