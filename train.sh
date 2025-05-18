#!/bin/bash

# Train RegNetY
python train_baseline.py --model RegNetY_400MF \
    --data-folder ./data \
    --batch_size 64 \
    --epochs 240 \
    --learning_rate 0.1 \
    --checkpoint-dir ./output-new-baseline-4090 &

# Train RegNetX
python train_baseline.py --model RegNetX_400MF \
    --data-folder ./data \
    --batch_size 64 \
    --epochs 240 \
    --learning_rate 0.1 \
    --checkpoint-dir ./output-new-baseline-4090 &

# Train ResNet
python train_baseline.py --model resnet32x4 \
    --data-folder ./data \
    --batch_size 64 \
    --epochs 240 \
    --learning_rate 0.1 \
    --checkpoint-dir ./output-new-baseline-4090 &

# Train WRN
python train_baseline.py --model wrn_28_4 \
    --data-folder ./data \
    --batch_size 64 \
    --epochs 240 \
    --learning_rate 0.1 \
    --checkpoint-dir ./output-new-baseline-4090 &

# Wait for all background processes to complete
wait 
