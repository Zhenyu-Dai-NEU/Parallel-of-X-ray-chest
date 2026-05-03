"""
Single-GPU Training with Parallel Optimization Benchmarks.
Compares: FP32 vs FP16, different num_workers, training throughput.
"""

import os
import time
import argparse
import json

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score
import numpy as np

from model import get_model, count_parameters
from dataset import get_dataloaders, DISEASE_LABELS


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, use_amp=False, scaler=None):
    model.train()
    total_loss = 0.0
    num_batches = 0
    num_samples = 0
    epoch_start = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        num_samples += images.size(0)

        if batch_idx % 50 == 0:
            elapsed = time.time() - epoch_start
            throughput = num_samples / elapsed if elapsed > 0 else 0
            gpu_mem = torch.cuda.memory_allocated(device) / 1e9
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Throughput: {throughput:.0f} img/s | "
                  f"GPU Mem: {gpu_mem:.2f}GB")

    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / num_batches
    throughput = num_samples / epoch_time
    return avg_loss, epoch_time, throughput


def evaluate(model, loader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if use_amp:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    aucs = {}
    for i, label_name in enumerate(DISEASE_LABELS):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            aucs[label_name] = auc
        except ValueError:
            aucs[label_name] = 0.0

    mean_auc = np.mean(list(aucs.values()))
    return avg_loss, mean_auc, aucs


def benchmark_num_workers(data_dir, csv_path, batch_size, img_size, device):
    """Benchmark different num_workers settings for DataLoader."""
    print(f"\n{'='*60}")
    print("Benchmark: DataLoader num_workers")
    print(f"{'='*60}")

    model = get_model(num_classes=14, pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    results = {}

    for nw in [0, 1, 2, 4, 8]:
        print(f"\nTesting num_workers={nw}...")
        train_loader, _, _, _ = get_dataloaders(
            data_dir, csv_path, batch_size=batch_size,
            num_workers=nw, img_size=img_size,
            use_processed=False, distributed=False,
        )

        # Warm up
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            if i >= 5:
                break

        # Benchmark: time 50 batches
        model.train()
        torch.cuda.synchronize()
        start = time.time()
        num_samples = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            num_samples += images.size(0)
            if i >= 49:
                break
        torch.cuda.synchronize()
        elapsed = time.time() - start

        throughput = num_samples / elapsed
        results[nw] = {"time_50_batches": elapsed, "throughput": throughput}
        print(f"  num_workers={nw}: {elapsed:.2f}s for 50 batches, {throughput:.0f} img/s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Single-GPU Training with Benchmarks")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--csv_path", type=str, default="./data/Data_Entry_2017_v2020.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--use_processed", action="store_true")
    parser.add_argument("--skip_benchmark", action="store_true", help="Skip num_workers benchmark")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Part 1: DataLoader num_workers benchmark
    if not args.skip_benchmark:
        workers_results = benchmark_num_workers(
            args.data_dir, args.csv_path, args.batch_size, args.img_size, device
        )
    else:
        workers_results = {}

    # Part 2: FP32 Training
    print(f"\n{'='*60}")
    print("Training: FP32 (baseline)")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader, _ = get_dataloaders(
        args.data_dir, args.csv_path, batch_size=args.batch_size,
        num_workers=args.num_workers, img_size=args.img_size,
        use_processed=args.use_processed, distributed=False,
    )

    model_fp32 = get_model(num_classes=14, pretrained=True).to(device)
    total_params, trainable_params = count_parameters(model_fp32)
    print(f"Model: DenseNet-121 | Params: {total_params:,} | Trainable: {trainable_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model_fp32.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    fp32_metrics = {"train_loss": [], "val_auc": [], "epoch_times": [],
                    "throughputs": [], "gpu_memory": []}
    total_fp32_time = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- FP32 Epoch {epoch}/{args.epochs} ---")
        torch.cuda.reset_peak_memory_stats()

        train_loss, epoch_time, throughput = train_one_epoch(
            model_fp32, train_loader, criterion, optimizer, device, epoch,
            use_amp=False
        )
        val_loss, val_auc, _ = evaluate(model_fp32, val_loader, criterion, device)
        scheduler.step()

        peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
        total_fp32_time += epoch_time

        fp32_metrics["train_loss"].append(train_loss)
        fp32_metrics["val_auc"].append(val_auc)
        fp32_metrics["epoch_times"].append(epoch_time)
        fp32_metrics["throughputs"].append(throughput)
        fp32_metrics["gpu_memory"].append(peak_mem)

        print(f"  Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | "
              f"Time: {epoch_time:.1f}s | Throughput: {throughput:.0f} img/s | "
              f"Peak GPU: {peak_mem:.2f}GB")

    # FP32 test
    fp32_test_loss, fp32_test_auc, fp32_test_aucs = evaluate(
        model_fp32, test_loader, criterion, device
    )
    print(f"\nFP32 Test AUC: {fp32_test_auc:.4f}")

    # Part 3: FP16 Mixed Precision Training
    print(f"\n{'='*60}")
    print("Training: FP16 Mixed Precision")
    print(f"{'='*60}")

    model_fp16 = get_model(num_classes=14, pretrained=True).to(device)
    optimizer_fp16 = Adam(model_fp16.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler_fp16 = CosineAnnealingLR(optimizer_fp16, T_max=args.epochs)
    scaler = GradScaler()

    fp16_metrics = {"train_loss": [], "val_auc": [], "epoch_times": [],
                    "throughputs": [], "gpu_memory": []}
    total_fp16_time = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- FP16 Epoch {epoch}/{args.epochs} ---")
        torch.cuda.reset_peak_memory_stats()

        train_loss, epoch_time, throughput = train_one_epoch(
            model_fp16, train_loader, criterion, optimizer_fp16, device, epoch,
            use_amp=True, scaler=scaler
        )
        val_loss, val_auc, _ = evaluate(model_fp16, val_loader, criterion, device, use_amp=True)
        scheduler_fp16.step()

        peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
        total_fp16_time += epoch_time

        fp16_metrics["train_loss"].append(train_loss)
        fp16_metrics["val_auc"].append(val_auc)
        fp16_metrics["epoch_times"].append(epoch_time)
        fp16_metrics["throughputs"].append(throughput)
        fp16_metrics["gpu_memory"].append(peak_mem)

        print(f"  Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | "
              f"Time: {epoch_time:.1f}s | Throughput: {throughput:.0f} img/s | "
              f"Peak GPU: {peak_mem:.2f}GB")

    # FP16 test
    fp16_test_loss, fp16_test_auc, fp16_test_aucs = evaluate(
        model_fp16, test_loader, criterion, device, use_amp=True
    )
    print(f"\nFP16 Test AUC: {fp16_test_auc:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'FP32':<15} {'FP16':<15} {'Speedup':<10}")
    print(f"{'-'*70}")
    fp32_avg = total_fp32_time / args.epochs
    fp16_avg = total_fp16_time / args.epochs
    speedup = fp32_avg / fp16_avg if fp16_avg > 0 else 0
    print(f"{'Avg epoch time (s)':<30} {fp32_avg:<15.2f} {fp16_avg:<15.2f} {speedup:<10.2f}x")
    print(f"{'Avg throughput (img/s)':<30} {np.mean(fp32_metrics['throughputs']):<15.0f} {np.mean(fp16_metrics['throughputs']):<15.0f}")
    print(f"{'Peak GPU memory (GB)':<30} {max(fp32_metrics['gpu_memory']):<15.2f} {max(fp16_metrics['gpu_memory']):<15.2f}")
    print(f"{'Test AUC':<30} {fp32_test_auc:<15.4f} {fp16_test_auc:<15.4f}")
    print(f"{'Total training time (s)':<30} {total_fp32_time:<15.2f} {total_fp16_time:<15.2f}")

    # Save all results
    results = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "workers_benchmark": workers_results,
        "fp32": {
            "total_time": total_fp32_time,
            "avg_epoch_time": fp32_avg,
            "test_auc": fp32_test_auc,
            "test_aucs": fp32_test_aucs,
            "metrics": fp32_metrics,
        },
        "fp16": {
            "total_time": total_fp16_time,
            "avg_epoch_time": fp16_avg,
            "test_auc": fp16_test_auc,
            "test_aucs": fp16_test_aucs,
            "metrics": fp16_metrics,
        },
        "speedup_fp16_over_fp32": speedup,
    }
    with open(os.path.join(args.output_dir, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/training_results.json")


if __name__ == "__main__":
    main()
