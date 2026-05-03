# Power of Parallel Computing in Thoracic Disease Classification
## Using NIH Chest X-ray Dataset on NEU Discovery Cluster

### Project Structure
```
chest_xray_parallel/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── download_data.sh             # Download NIH ChestX-ray14 dataset
├── preprocess_parallel.py       # Parallel data preprocessing (multiprocessing)
├── dataset.py                   # PyTorch Dataset class
├── model.py                     # DenseNet-121 model definition
├── train_single.py              # Single-GPU training (FP32 vs FP16 benchmark)
├── multigpu.ipynb               # Multi-GPU experiments (DDP & FSDP)
├── benchmark.py                 # Visualization of benchmark results
├── submit_preprocess.sh         # SLURM job script - parallel preprocessing
└── submit_train.sh              # SLURM job script - single-GPU training
```

### Quick Start on NEU Discovery

```bash
# 1. Login to Discovery
ssh <your_username>@login.discovery.neu.edu

# 2. Clone/upload this project
cd /scratch/<your_username>/
# Upload files here

# 3. Load modules
module load anaconda3/2022.05
module load cuda/11.8

# 4. Create conda environment
conda create -n chest_xray python=3.10 -y
conda activate chest_xray
pip install -r requirements.txt

# 5. Download dataset and run parallel preprocessing benchmark
sbatch submit_preprocess.sh  # or run interactively

# 6. Run single-GPU training (FP32 vs FP16 benchmark)
sbatch submit_train.sh

# 7. Run multi-GPU experiments (DDP & FSDP)
#    Launch Jupyter on a GPU node and open multigpu.ipynb
```

### Experiments Covered

| Level | Strategy | File |
|-------|----------|------|
| CPU data preprocessing | `multiprocessing.Pool` (1/2/4/8/16 workers) | `preprocess_parallel.py` |
| DataLoader parallelism | PyTorch `num_workers` sweep | `train_single.py` |
| Single-GPU training | FP32 vs FP16 mixed precision | `train_single.py` |
| Multi-GPU training | DDP and FSDP | `multigpu.ipynb` |
| Visualization | Speedup / efficiency / throughput charts | `benchmark.py` |
