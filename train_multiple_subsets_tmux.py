import subprocess
import time
from dataset.cifiar100_subset import create_cifar100_subset_dataloaders,get_cifar100_subset_class
# 配置
model_list = [
    "resnet20",
    "resnet56",
    "RegNetY_400MF",
]

# overlap 比例（如 0.2 表示 20% overlap）
overlaps = [0.5]


n = len(model_list)
num_classes = 100


CIFAR100_CLASSES = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
        'bed', 'bee', 'beetle', 'bicycle', 'bottle',
        'bowl', 'boy', 'bridge', 'bus', 'butterfly',
        'camel', 'can', 'castle', 'caterpillar', 'cattle',
        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
        'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox',
        'girl', 'hamster', 'house', 'kangaroo', 'computer_keyboard',
        'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
        'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
        'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
        'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
        'plain', 'plate', 'poppy', 'porcupine', 'possum',
        'rabbit', 'raccoon', 'ray', 'road', 'rocket',
        'rose', 'sea', 'seal', 'shark', 'shrew',
        'skunk', 'skyscraper', 'snail', 'snake', 'spider',
        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor',
        'train', 'trout', 'tulip', 'turtle', 'wardrobe',
        'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]

def tmux_session_exists(session_name):
    result = subprocess.run(f"tmux has-session -t {session_name}", shell=True)
    return result.returncode == 0

for overlap in overlaps:
    print(f"Training with overlap: {overlap}")
    # 创建数据加载器

    data_folder = "./data"
    checkpoint_dir = f"./baseline-output-subset-overlap-{overlap}"
    gpu_ids = ["0"]  # 按需分配"
    trial = "0"

    session_name = f"overlap-{int(overlap*10)}_subset_baseline_train"
    print(f"Creating tmux session: {session_name}")
    # 创建 tmux session
    subprocess.run(f"tmux new-session -d -s {session_name}", shell=True)
    # 等待 session ready
    for _ in range(20):
        if tmux_session_exists(session_name):
            break
        time.sleep(1)
    else:
        print(f"Session {session_name} not created!")
        continue

    # 生成所有子集的类别索引，严格保证 overlap
    # 子集数量与模型数量相等，严格保证 overlap
    subsets = get_cifar100_subset_class(overlap, len(model_list), num_classes)
    for idx, class_list in enumerate(subsets):
        model = model_list[idx]
        class_list_str = ",".join(str(c) for c in class_list)
        class_names = [CIFAR100_CLASSES[c] for c in class_list]
        class_names_str = ", ".join(class_names)
        print(f"Training model: {model}, Classes: {class_list_str}")
        print(f"Class names: {class_names_str}")

        window_name = f"{model}_{idx}"
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu_id} python train_baseline.py "
            f"--model {model} "
            f"--class-list {class_list_str} "
            f"--data-folder {data_folder} "
            f"--checkpoint-dir {checkpoint_dir} "
            f"--gpu_id {gpu_id} "
            f"--trial {trial}"
        )
        if idx == 0:
            subprocess.run(f"tmux rename-window -t {session_name}:0 {window_name}", shell=True)
            subprocess.run(f"tmux send-keys -t {session_name}:0 '{cmd}' C-m", shell=True)
        else:
            subprocess.run(f"tmux new-window -t {session_name} -n {window_name}", shell=True)
            subprocess.run(f"tmux send-keys -t {session_name}:{window_name} '{cmd}' C-m", shell=True)
        time.sleep(0.2)

    print(f"All subset training started in tmux session '{session_name}'.")
    print("Attach with: tmux attach-session -t", session_name) 
