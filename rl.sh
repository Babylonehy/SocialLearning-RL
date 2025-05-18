python train_student_rl.py \
    --data /home/tju/Projects/data\
    --arch ShuffleV2 \
    --dynamic \
    --checkpoint-dir ./mtkd-rl-output \
    --teacher-name-list RegNetY_400MF RegNetX_400MF resnet32x4 wrn_28_4 \
    --dist-backend 'nccl' \
    --world-size 1 \
    --rank 0 
    
python train_student_rl.py \
    --data /home/tju/Projects/data\
    --arch MobileNetV2 \
    --dynamic \
    --checkpoint-dir ./mtkd-rl-output \
    --teacher-name-list RegNetY_400MF RegNetX_400MF resnet32x4 wrn_28_4 \
    --dist-backend 'nccl' \
    --world-size 1 \
    --rank 0 

NCCL_P2P_LEVEL=NVL python train_student_rl.py \
    --data /mnt/cifar \
    --arch ShuffleV2 \
    --dynamic \
    --batch-size 32 \
    --checkpoint-dir ./mtkd-rl-output-8 \
    --teacher-name-list RegNetY_400MF RegNetX_400MF resnet32x4 wrn_28_4 \
    --dist-backend 'nccl' \
    --world-size 1 \
    --dist-url tcp://127.0.0.1:23466 \
    --multiprocessing-distributed \
    --rank 0 \
    --gpu 0 

NCCL_P2P_LEVEL=NVL torchrun --nproc_per_node=4 train_student_rl_torchrun.py \
    --batch-size 64 \
    --data /mnt/cifar \
    --arch ShuffleV2 \
    --checkpoint-dir ./mtkd-rl-output-8 \
    --dynamic \
    --teacher-name-list RegNetY_400MF RegNetX_400MF resnet32x4 wrn_28_4 

NCCL_P2P_LEVEL=NVL torchrun --nproc_per_node=8 train_student_rl.py \
    --batch-size 64 \
    --data /mnt/cifar \
    --arch ShuffleV2 \
    --checkpoint-dir ./mtkd-rl-output-8 \
    --dynamic \
    --teacher-name-list RegNetY_400MF RegNetX_400MF resnet32x4 wrn_28_4 \
    --dist-backend 'nccl' \
    --multiprocessing-distributed 
