"""
Benchmark Visualization Script.
Generates comparison charts from preprocessing and training results.
"""

import os
import json
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_preprocessing_benchmark(preprocess_results, output_dir):
    """Plot CPU multiprocessing speedup for data preprocessing."""
    if not preprocess_results:
        print("No preprocessing results found, skipping...")
        return

    workers = sorted([int(k) for k in preprocess_results.keys()])
    times = [preprocess_results[str(w)]["time"] for w in workers]
    speedups = [times[0] / t for t in times]
    efficiencies = [s / w * 100 if w > 0 else 100 for s, w in zip(speedups, workers)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Time vs Workers
    ax = axes[0]
    ax.bar(range(len(workers)), times, color="#2196F3", tick_label=[str(w) for w in workers])
    ax.set_xlabel("Number of CPU Workers")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Preprocessing Time vs CPU Workers")
    ax.grid(True, alpha=0.3, axis="y")

    # Speedup
    ax = axes[1]
    ax.plot(workers, speedups, "o-", color="#4CAF50", linewidth=2, markersize=8, label="Actual")
    ax.plot(workers, workers, "--", color="gray", alpha=0.5, label="Ideal (linear)")
    ax.set_xlabel("Number of CPU Workers")
    ax.set_ylabel("Speedup (x)")
    ax.set_title("Preprocessing Speedup")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Efficiency
    ax = axes[2]
    ax.bar(range(len(workers)), efficiencies, color="#FF9800", tick_label=[str(w) for w in workers])
    ax.axhline(y=100, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Number of CPU Workers")
    ax.set_ylabel("Efficiency (%)")
    ax.set_title("Parallel Efficiency")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "preprocessing_benchmark.png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir}/preprocessing_benchmark.png")


def plot_workers_benchmark(workers_results, output_dir):
    """Plot DataLoader num_workers comparison."""
    if not workers_results:
        print("No workers benchmark results, skipping...")
        return

    workers = sorted([int(k) for k in workers_results.keys()])
    throughputs = [workers_results[str(w)]["throughput"] for w in workers]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(workers)), throughputs, color="#2196F3",
                  tick_label=[str(w) for w in workers])
    ax.set_xlabel("num_workers")
    ax.set_ylabel("Throughput (images/sec)")
    ax.set_title("DataLoader Throughput vs num_workers")
    for bar, tp in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                f"{tp:.0f}", ha="center", va="bottom", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "workers_benchmark.png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir}/workers_benchmark.png")


def plot_fp32_vs_fp16(results, output_dir):
    """Plot FP32 vs FP16 training comparison."""
    fp32 = results.get("fp32", {})
    fp16 = results.get("fp16", {})
    if not fp32 or not fp16:
        print("Missing FP32 or FP16 results, skipping...")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(fp32["metrics"]["train_loss"]) + 1)

    # Training Loss
    ax = axes[0][0]
    ax.plot(epochs, fp32["metrics"]["train_loss"], "o-", color="#2196F3", label="FP32", linewidth=2)
    ax.plot(epochs, fp16["metrics"]["train_loss"], "s-", color="#FF9800", label="FP16", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss: FP32 vs FP16")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Validation AUC
    ax = axes[0][1]
    ax.plot(epochs, fp32["metrics"]["val_auc"], "o-", color="#2196F3", label="FP32", linewidth=2)
    ax.plot(epochs, fp16["metrics"]["val_auc"], "s-", color="#FF9800", label="FP16", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUC")
    ax.set_title("Validation AUC: FP32 vs FP16")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Throughput
    ax = axes[1][0]
    ax.plot(epochs, fp32["metrics"]["throughputs"], "o-", color="#2196F3", label="FP32", linewidth=2)
    ax.plot(epochs, fp16["metrics"]["throughputs"], "s-", color="#FF9800", label="FP16", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Throughput (images/sec)")
    ax.set_title("Training Throughput: FP32 vs FP16")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # GPU Memory
    ax = axes[1][1]
    x = np.arange(2)
    fp32_mem = max(fp32["metrics"]["gpu_memory"])
    fp16_mem = max(fp16["metrics"]["gpu_memory"])
    bars = ax.bar(x, [fp32_mem, fp16_mem], color=["#2196F3", "#FF9800"], width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["FP32", "FP16"])
    ax.set_ylabel("Peak GPU Memory (GB)")
    ax.set_title("Peak GPU Memory Usage")
    for bar, mem in zip(bars, [fp32_mem, fp16_mem]):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                f"{mem:.2f}GB", ha="center", va="bottom", fontweight="bold")
    savings = (1 - fp16_mem / fp32_mem) * 100 if fp32_mem > 0 else 0
    ax.set_title(f"Peak GPU Memory (FP16 saves {savings:.0f}%)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fp32_vs_fp16.png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir}/fp32_vs_fp16.png")

    # Summary bar chart
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

    # Speedup
    ax = axes2[0]
    speedup = fp32["avg_epoch_time"] / fp16["avg_epoch_time"] if fp16["avg_epoch_time"] > 0 else 0
    bars = ax.bar(["FP32", "FP16"], [1.0, speedup], color=["#2196F3", "#FF9800"], width=0.5)
    ax.set_ylabel("Relative Speed")
    ax.set_title(f"FP16 Speedup: {speedup:.2f}x")
    ax.grid(True, alpha=0.3, axis="y")

    # Time comparison
    ax = axes2[1]
    bars = ax.bar(["FP32", "FP16"], [fp32["total_time"], fp16["total_time"]],
                  color=["#2196F3", "#FF9800"], width=0.5)
    ax.set_ylabel("Total Training Time (s)")
    ax.set_title("Total Training Time")
    for bar, t in zip(bars, [fp32["total_time"], fp16["total_time"]]):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                f"{t:.0f}s", ha="center", va="bottom")
    ax.grid(True, alpha=0.3, axis="y")

    # AUC comparison
    ax = axes2[2]
    bars = ax.bar(["FP32", "FP16"], [fp32["test_auc"], fp16["test_auc"]],
                  color=["#2196F3", "#FF9800"], width=0.5)
    ax.set_ylabel("Test AUC")
    ax.set_title("Model Quality (Test AUC)")
    ax.set_ylim(0, 1)
    for bar, auc in zip(bars, [fp32["test_auc"], fp16["test_auc"]]):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f"{auc:.4f}", ha="center", va="bottom")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_summary.png"), dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir}/training_summary.png")


def main():
    parser = argparse.ArgumentParser(description="Generate Benchmark Charts")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--output_dir", type=str, default="./results/charts")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load training results
    training_path = os.path.join(args.results_dir, "training_results.json")
    if os.path.exists(training_path):
        with open(training_path) as f:
            training_results = json.load(f)
        print("Loaded training results")
        plot_workers_benchmark(training_results.get("workers_benchmark", {}), args.output_dir)
        plot_fp32_vs_fp16(training_results, args.output_dir)
    else:
        print(f"Warning: {training_path} not found")

    # Load preprocessing results
    preprocess_path = os.path.join(args.results_dir, "preprocess_results.json")
    if os.path.exists(preprocess_path):
        with open(preprocess_path) as f:
            preprocess_results = json.load(f)
        print("Loaded preprocessing results")
        plot_preprocessing_benchmark(preprocess_results, args.output_dir)
    else:
        print(f"Warning: {preprocess_path} not found")

    print(f"\nAll charts saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
