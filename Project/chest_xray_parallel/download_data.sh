#!/bin/bash
# Download NIH ChestX-ray14 dataset
# Run this on a compute node with internet access, or use Kaggle API

# Option 1: Using Kaggle API
# pip install kaggle
# export KAGGLE_USERNAME=<your_username>
# export KAGGLE_KEY=<your_key>
# kaggle datasets download -d nih-chest-xrays/data -p ./data/

# Option 2: Manual download
# Download from https://www.kaggle.com/datasets/nih-chest-xrays/data
# Place files in ./data/ directory

DATA_DIR="./data"
mkdir -p $DATA_DIR

echo "============================================"
echo "NIH ChestX-ray14 Dataset Download"
echo "============================================"
echo ""
echo "Dataset size: ~42 GB"
echo "Target directory: $DATA_DIR"
echo ""
echo "Please download the dataset from:"
echo "  https://www.kaggle.com/datasets/nih-chest-xrays/data"
echo ""
echo "Expected structure after download:"
echo "  data/"
echo "  ├── images_001/ to images_012/  (PNG files)"
echo "  ├── Data_Entry_2017_v2020.csv   (labels)"
echo "  ├── train_val_list.txt"
echo "  └── test_list.txt"
echo ""
echo "If using Kaggle CLI:"
echo "  pip install kaggle"
echo "  kaggle datasets download -d nih-chest-xrays/data -p $DATA_DIR"
