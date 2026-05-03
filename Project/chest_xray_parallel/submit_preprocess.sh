#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=courses-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

mkdir -p logs results data/processed

module load anaconda3/2024.06
module load cuda/12.1.1
source activate chest_xray

echo "============================================"
echo "CPU Parallel Preprocessing Benchmark"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "============================================"

python preprocess_parallel.py \
    --data_dir ./data \
    --output_dir ./data/processed \
    --csv_path ./data/Data_Entry_2017_v2020.csv \
    --target_size 224 \
    --max_images 2000 \
    --results_dir ./results

echo "Preprocessing benchmark complete!"
