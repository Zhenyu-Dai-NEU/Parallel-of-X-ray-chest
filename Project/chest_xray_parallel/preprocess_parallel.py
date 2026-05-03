"""
Parallel Data Preprocessing for NIH ChestX-ray14 Dataset.
Benchmarks sequential vs parallel (multiprocessing) across different CPU core counts.
"""

import os
import time
import argparse
import json
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def process_single_image(args, src_dir, dst_dir, target_size):
    """Process a single image: resize and save."""
    img_name, labels = args
    src_path = find_image(img_name, src_dir)
    if src_path is None:
        return None

    try:
        img = Image.open(src_path).convert("RGB")
        img = img.resize((target_size, target_size), Image.BILINEAR)
        dst_path = os.path.join(dst_dir, img_name)
        img.save(dst_path)
        return {"image": img_name, "labels": labels, "status": "ok"}
    except Exception as e:
        return {"image": img_name, "labels": labels, "status": f"error: {e}"}


def find_image(img_name, src_dir):
    """Find image across multiple subdirectories."""
    direct = os.path.join(src_dir, img_name)
    if os.path.exists(direct):
        return direct
    for subdir in sorted(Path(src_dir).iterdir()):
        if subdir.is_dir() and subdir.name.startswith("images_"):
            for candidate in [subdir / "images" / img_name, subdir / img_name]:
                if candidate.exists():
                    return str(candidate)
    return None


def preprocess_sequential(image_label_pairs, src_dir, dst_dir, target_size):
    """Sequential preprocessing baseline."""
    results = []
    for pair in tqdm(image_label_pairs, desc="Sequential"):
        result = process_single_image(pair, src_dir, dst_dir, target_size)
        if result:
            results.append(result)
    return results


def preprocess_parallel(image_label_pairs, src_dir, dst_dir, target_size, num_workers):
    """Parallel preprocessing using multiprocessing Pool."""
    func = partial(process_single_image, src_dir=src_dir, dst_dir=dst_dir, target_size=target_size)
    results = []
    with Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(func, image_label_pairs, chunksize=100),
            total=len(image_label_pairs),
            desc=f"Parallel ({num_workers} workers)",
        ):
            if result:
                results.append(result)
    return results


def load_labels(csv_path):
    """Load and parse the label CSV file."""
    df = pd.read_csv(csv_path)
    image_label_pairs = []
    for _, row in df.iterrows():
        img_name = row["Image Index"]
        labels = row["Finding Labels"]
        image_label_pairs.append((img_name, labels))
    return image_label_pairs


def main():
    parser = argparse.ArgumentParser(description="Parallel Preprocessing Benchmark")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./data/processed")
    parser.add_argument("--csv_path", type=str, default="./data/Data_Entry_2017_v2020.csv")
    parser.add_argument("--target_size", type=int, default=224)
    parser.add_argument("--max_images", type=int, default=2000, help="Number of images for benchmark")
    parser.add_argument("--results_dir", type=str, default="./results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Loading labels from {args.csv_path}...")
    image_label_pairs = load_labels(args.csv_path)
    print(f"Total images in dataset: {len(image_label_pairs)}")

    # Use subset for benchmark
    subset = image_label_pairs[:args.max_images]
    print(f"Using {len(subset)} images for benchmark")
    print(f"Available CPU cores: {mp.cpu_count()}")

    benchmark_results = {}

    # Sequential baseline
    print(f"\n{'='*60}")
    print("Sequential Preprocessing (1 worker)")
    print(f"{'='*60}")
    t0 = time.time()
    preprocess_sequential(subset, args.data_dir, args.output_dir, args.target_size)
    seq_time = time.time() - t0
    print(f"Time: {seq_time:.2f}s | Throughput: {len(subset)/seq_time:.0f} img/s")
    benchmark_results["1"] = {
        "time": seq_time,
        "throughput": len(subset) / seq_time,
        "speedup": 1.0,
        "efficiency": 100.0,
    }

    # Parallel with different worker counts
    worker_counts = [2, 4, 8, 16]
    worker_counts = [w for w in worker_counts if w <= mp.cpu_count()]

    for nw in worker_counts:
        print(f"\n{'='*60}")
        print(f"Parallel Preprocessing ({nw} workers)")
        print(f"{'='*60}")
        t0 = time.time()
        preprocess_parallel(subset, args.data_dir, args.output_dir, args.target_size, nw)
        par_time = time.time() - t0
        speedup = seq_time / par_time
        efficiency = speedup / nw * 100
        print(f"Time: {par_time:.2f}s | Throughput: {len(subset)/par_time:.0f} img/s")
        print(f"Speedup: {speedup:.2f}x | Efficiency: {efficiency:.1f}%")
        benchmark_results[str(nw)] = {
            "time": par_time,
            "throughput": len(subset) / par_time,
            "speedup": speedup,
            "efficiency": efficiency,
        }

    # Print summary
    print(f"\n{'='*60}")
    print("PREPROCESSING BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Workers':<10} {'Time(s)':<12} {'Throughput':<15} {'Speedup':<10} {'Efficiency':<10}")
    print(f"{'-'*57}")
    for w in ["1"] + [str(w) for w in worker_counts]:
        r = benchmark_results[w]
        print(f"{w:<10} {r['time']:<12.2f} {r['throughput']:<15.0f} {r['speedup']:<10.2f}x {r['efficiency']:<10.1f}%")

    # Save results
    results_path = os.path.join(args.results_dir, "preprocess_results.json")
    with open(results_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
