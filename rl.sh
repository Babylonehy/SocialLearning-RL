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

python train_multi_student_rl.py \
    --data /home/tju/Projects/data \
    --student-arch-list resnet56 resnet20 \
    --teacher-name-list RegNetY_400MF RegNetX_400MF resnet32x4 wrn_28_4 \
    --batch-size 64 \
    --epochs 240 \
    --lr 0.05 \
    --agent-lr 0.001 \
    --teacher-kd-weight 1.0 \
    --teacher-feat-weight 1.0 \
    --mutual-kd-weight 0.5 \
    --mutual-feat-weight 0.25 \
    --ce-weight 1.0 \
    --dynamic \
    --print-freq 10 \
    --checkpoint-dir ./checkpoints_multi_student_test \
    --trial test_run_01 \
    --seed 42 \
    --gpu 0

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
