import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


NUM_HOSTS = 6
HOST_CAPACITY = 60.0
NUM_VMS = 18
TIME_STEPS = 50
SHIFT_TIME = 25
TRAIN_STEPS = 15
SEED = 7


def generate_workload():
    """Create repeatable VM CPU demands with a clear shift halfway through."""
    rng = np.random.default_rng(SEED)
    base_loads = rng.uniform(8, 18, NUM_VMS)
    phases = rng.uniform(0, 2 * np.pi, NUM_VMS)
    workload = np.zeros((TIME_STEPS, NUM_VMS))

    for time in range(TIME_STEPS):
        for vm in range(NUM_VMS):
            seasonal = 1 + 0.12 * np.sin(2 * np.pi * time / 10 + phases[vm])
            shift = 1.0
            if time >= SHIFT_TIME:
                progress = min((time - SHIFT_TIME + 1) / 8, 1.0)
                shift += progress * (0.80 if vm % 3 == 0 else 0.20)
            noise = rng.normal(0, 0.6)
            workload[time, vm] = max(2.0, base_loads[vm] * seasonal * shift + noise)

    return workload


def first_fit(vm_loads):
    assignment = [-1] * len(vm_loads)
    host_loads = [0.0] * NUM_HOSTS

    for vm, load in enumerate(vm_loads):
        for host in range(NUM_HOSTS):
            if host_loads[host] + load <= HOST_CAPACITY:
                assignment[vm] = host
                host_loads[host] += load
                break
        if assignment[vm] == -1:
            host = int(np.argmin(host_loads))
            assignment[vm] = host
            host_loads[host] += load

    return assignment


def best_fit(vm_loads):
    assignment = [-1] * len(vm_loads)
    host_loads = [0.0] * NUM_HOSTS

    for vm, load in enumerate(vm_loads):
        fitting_hosts = [
            host
            for host in range(NUM_HOSTS)
            if host_loads[host] + load <= HOST_CAPACITY
        ]
        if fitting_hosts:
            host = min(
                fitting_hosts,
                key=lambda item: HOST_CAPACITY - host_loads[item] - load,
            )
        else:
            host = int(np.argmin(host_loads))
        assignment[vm] = host
        host_loads[host] += load

    return assignment


def train_predictor(workload, current_time):
    """Train on known history using current, previous, and 3-step mean load."""
    features = []
    targets = []
    for time in range(2, current_time):
        for vm in range(NUM_VMS):
            features.append(
                [
                    workload[time, vm],
                    workload[time - 1, vm],
                    workload[time - 2 : time + 1, vm].mean(),
                ]
            )
            targets.append(workload[time + 1, vm])

    return LinearRegression().fit(features, targets)


def predict_loads(model, workload, current_time):
    features = []
    for vm in range(NUM_VMS):
        features.append(
            [
                workload[current_time, vm],
                workload[current_time - 1, vm],
                workload[current_time - 2 : current_time + 1, vm].mean(),
            ]
        )
    return np.maximum(model.predict(features), 2.0)


def predictive_placement(assignment, predicted_loads):
    """Move large VMs away from hosts predicted to exceed capacity."""
    assignment = assignment.copy()
    migrations = 0

    while True:
        host_loads = np.zeros(NUM_HOSTS)
        for vm, host in enumerate(assignment):
            host_loads[host] += predicted_loads[vm]

        overloaded_hosts = [
            host for host in range(NUM_HOSTS) if host_loads[host] > HOST_CAPACITY
        ]
        moved = False
        for source in overloaded_hosts:
            source_vms = sorted(
                (vm for vm, host in enumerate(assignment) if host == source),
                key=lambda vm: predicted_loads[vm],
                reverse=True,
            )
            for vm in source_vms:
                destinations = [
                    host
                    for host in range(NUM_HOSTS)
                    if host != source
                    and host_loads[host] + predicted_loads[vm] <= HOST_CAPACITY
                ]
                if destinations:
                    assignment[vm] = min(destinations, key=lambda host: host_loads[host])
                    migrations += 1
                    moved = True
                    break
            if moved:
                break
        if not moved:
            break

    return assignment, migrations


def calculate_metrics(assignment, actual_loads, migrations):
    host_loads = np.zeros(NUM_HOSTS)
    for vm, host in enumerate(assignment):
        host_loads[host] += actual_loads[vm]

    return {
        "utilization_imbalance": float(np.std(host_loads / HOST_CAPACITY)),
        "sla_violations": int(np.sum(host_loads > HOST_CAPACITY)),
        "overload_cpu": float(np.maximum(host_loads - HOST_CAPACITY, 0).sum()),
        "migrations": migrations,
    }


def run_experiment(workload):
    rows = []
    first_fit_assignment = first_fit(workload[0])
    best_fit_assignment = best_fit(workload[0])
    predictive_assignment = best_fit_assignment.copy()

    for time in range(TIME_STEPS):
        predictive_migrations = 0
        if time >= TRAIN_STEPS:
            # Targets used for training end at time - 1; time is never seen.
            model = train_predictor(workload, time - 1)
            predicted_loads = predict_loads(model, workload, time - 1)
            predictive_assignment, predictive_migrations = predictive_placement(
                predictive_assignment, predicted_loads
            )

        assignments = {
            "First-Fit": (first_fit_assignment, 0),
            "Best-Fit": (best_fit_assignment, 0),
            "Predictive Placement": (predictive_assignment, predictive_migrations),
        }
        for method, (assignment, migrations) in assignments.items():
            metrics = calculate_metrics(assignment, workload[time], migrations)
            rows.append(
                {
                    "time": time,
                    "phase": "before" if time < SHIFT_TIME else "after",
                    "method": method,
                    **metrics,
                }
            )

    return rows


def save_results(rows, output_dir):
    fieldnames = [
        "time",
        "phase",
        "method",
        "utilization_imbalance",
        "sla_violations",
        "overload_cpu",
        "migrations",
    ]
    with (output_dir / "time_series.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for method in ("First-Fit", "Best-Fit", "Predictive Placement"):
        method_rows = [row for row in rows if row["method"] == method]
        before_rows = [row for row in method_rows if row["phase"] == "before"]
        after_rows = [row for row in method_rows if row["phase"] == "after"]
        summary.append(
            {
                "method": method,
                "imbalance_before_shift": sum(
                    row["utilization_imbalance"] for row in before_rows
                )
                / len(before_rows),
                "imbalance_after_shift": sum(
                    row["utilization_imbalance"] for row in after_rows
                )
                / len(after_rows),
                "sla_violations_before_shift": sum(
                    row["sla_violations"] for row in before_rows
                ),
                "sla_violations_after_shift": sum(
                    row["sla_violations"] for row in after_rows
                ),
                "total_sla_violations": sum(
                    row["sla_violations"] for row in method_rows
                ),
                "total_overload_cpu": sum(row["overload_cpu"] for row in method_rows),
                "migration_count": sum(row["migrations"] for row in method_rows),
            }
        )

    with (output_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    return summary


def plot_results(rows, output_dir):
    methods = ("First-Fit", "Best-Fit", "Predictive Placement")
    figure, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        times = [row["time"] for row in method_rows]
        axes[0].plot(
            times,
            [row["utilization_imbalance"] for row in method_rows],
            label=method,
        )
        axes[1].plot(
            times, [row["sla_violations"] for row in method_rows], label=method
        )
        axes[2].plot(
            times,
            np.cumsum([row["migrations"] for row in method_rows]),
            label=method,
        )

    for axis in axes:
        axis.axvline(SHIFT_TIME, color="black", linestyle="--", label="Workload shift")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Utilization imbalance")
    axes[1].set_ylabel("SLA violations")
    axes[2].set_ylabel("Cumulative migrations")
    axes[2].set_xlabel("Time step")
    axes[0].legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "comparison.png", dpi=150)
    plt.close(figure)


def main():
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    workload = generate_workload()
    rows = run_experiment(workload)
    summary = save_results(rows, output_dir)
    plot_results(rows, output_dir)

    print("Method                 Imbalance before/after   SLA before/after/total   Migrations")
    for row in summary:
        print(
            f"{row['method']:<22}"
            f"{row['imbalance_before_shift']:.3f}/{row['imbalance_after_shift']:.3f}"
            f"             {row['sla_violations_before_shift']}/"
            f"{row['sla_violations_after_shift']}/{row['total_sla_violations']}"
            f"                    {row['migration_count']}"
        )
    print("\nSaved results in the results folder.")


if __name__ == "__main__":
    main()
