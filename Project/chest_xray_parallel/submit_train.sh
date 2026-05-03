#!/bin/bash
#SBATCH --job-name=train_gpu
#SBATCH --partition=courses-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

mkdir -p logs results

module load anaconda3/2024.06
module load cuda/12.1.1
source activate chest_xray

echo "============================================"
echo "GPU Training: FP32 vs FP16 Benchmark"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================"

python train_single.py \
    --data_dir ./data \
    --csv_path ./data/Data_Entry_2017_v2020.csv \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-4 \
    --num_workers 4 \
    --img_size 224 \
    --output_dir ./results

echo "Training complete!"

# Generate charts
python benchmark.py --results_dir ./results --output_dir ./results/charts

echo "Charts generated!"
