# import torch
# from torch.utils.data import DataLoader, Dataset, Subset
# from torchvision.datasets import CIFAR100
# from torchvision import transforms
# #cifar 100 class and class name
# # CIFAR100 class names
# # CIFAR100_CLASSES = [
# #     'apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
# #     'bed', 'bee', 'beetle', 'bicycle', 'bottle',

# class CIFAR100Subset(Dataset):
#     def __init__(self, base_dataset, class_list):
#         self.class_list = class_list
#         self.indices = [i for i, t in enumerate(base_dataset.targets) if t in class_list]
#         self.base_dataset = base_dataset

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         img, target = self.base_dataset[idx]
#         return img, target

# def get_cifar100_dataloaders_subset(class_list, batch_size=128, num_workers=8, root='./data'):
#     train_transform = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
#     ])
#     test_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
#     ])
#     trainset = CIFAR100(root=root, train=True, download=True, transform=train_transform)
#     testset = CIFAR100(root=root, train=False, download=True, transform=test_transform)

#     train_subset = CIFAR100Subset(trainset, class_list)
#     test_subset = CIFAR100Subset(testset, class_list)
#     # full_testset = CIFAR100Subset(testset)  # Full test set for validation
    
#     train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
#     # full_test_loader = DataLoader(full_testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
#     # Number of classes in CIFAR-100
#     num_classes = 100
#     # If class_list is provided, adjust num_classes accordingly
#     if class_list:
#         subset_num_classes = len(class_list)
#     # print the size of train and test set
#     print(f"Train set size: {len(train_subset)}")
#     print(f"Test set size: {len(test_subset)}")
#     return train_loader, test_loader, num_classes, subset_num_classes

# # 用法示例
# if __name__ == "__main__":
#     class_list = [10, 11, 12, 13]
#     train_loader, test_loader, num_classes = get_cifar100_dataloaders_subset(class_list)
#     print(f"num_classes: {num_classes}")
#     print(f"Number of training samples: {len(train_loader.dataset)}")
#     print(f"Number of test samples: {len(test_loader.dataset)}")
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
from typing import List, Optional, Tuple, Any

class CIFAR100Subset(Dataset):
    """
    CIFAR-100 子集数据集，只包含指定的类别，但保持原始的100类标签
    """
    
    def __init__(self, 
                 root: str = './data', 
                 train: bool = True, 
                 class_list: List[int] = None,
                 transform: Optional[transforms.Compose] = None,
                 target_transform: Optional[transforms.Compose] = None,
                 download: bool = True):
        """
        Args:
            root: 数据根目录
            train: 是否为训练集
            class_list: 要包含的类别列表，范围[0, 99]
            transform: 输入变换
            target_transform: 标签变换
            download: 是否下载数据
        """
        if class_list is None:
            class_list = list(range(100))  # 默认包含所有类别
        
        # 验证class_list的有效性
        if not all(0 <= c <= 99 for c in class_list):
            raise ValueError("class_list中的类别必须在[0, 99]范围内")
        
        self.class_list = set(class_list)
        self.transform = transform
        self.target_transform = target_transform
        
        # 加载原始CIFAR-100数据集
        self.cifar100 = datasets.CIFAR100(
            root=root, 
            train=train, 
            download=download,
            transform=None,  # 我们稍后手动应用变换
            target_transform=None
        )
        
        # 筛选出指定类别的数据
        self.filtered_indices = []
        for idx, (_, label) in enumerate(self.cifar100):
            if label in self.class_list:
                self.filtered_indices.append(idx)
        
        print(f"原始数据集大小: {len(self.cifar100)}")
        print(f"筛选后数据集大小: {len(self.filtered_indices)}")
        print(f"包含的类别: {sorted(self.class_list)}")
        
        # 获取类别名称
        self.class_names = self.cifar100.classes
    
    def __len__(self) -> int:
        return len(self.filtered_indices)
    
    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        if idx >= len(self.filtered_indices):
            raise IndexError("索引超出范围")
        
        # 获取原始数据集中的索引
        original_idx = self.filtered_indices[idx]
        image, label = self.cifar100[original_idx]
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
    
    def get_class_distribution(self) -> dict:
        """获取类别分布"""
        distribution = {}
        for idx in self.filtered_indices:
            _, label = self.cifar100[idx]
            distribution[label] = distribution.get(label, 0) + 1
        
        return distribution
    
    def get_class_names(self) -> List[str]:
        """获取所有类别名称"""
        return self.class_names
    
    def get_filtered_class_names(self) -> List[str]:
        """获取筛选后的类别名称"""
        return [self.class_names[i] for i in sorted(self.class_list)]

def create_category_groups(num_categories, num_groups, overlap_ratio):
    """
    改进版本：更清晰的循环分配逻辑
    
    Args:
        num_categories (int): 类别总数
        num_groups (int): 分组数量
        overlap_ratio (float): 重复比例
    
    Returns:
        list: 每个组的类别id列表
    """
    if num_groups <= 0 or num_categories <= 0:
        return []
    
    if num_groups == 1:
        return [list(range(num_categories))]
    
    # 计算相邻组重复的类别数量
    overlap_count = int(num_categories * overlap_ratio)
    overlap_count = max(0, min(overlap_count, num_categories))
    
    # 计算每组的有效步长（去除重复后的新增类别数）
    if overlap_count == 0:
        # 无重复情况，平均分配
        step_size = num_categories // num_groups
        remainder = num_categories % num_groups
    else:
        # 有重复情况，计算步长
        step_size = max(1, (num_categories - overlap_count) // num_groups)
    
    # 计算每组大小
    group_size = step_size + overlap_count
    
    groups = []
    
    for i in range(num_groups):
        start_pos = (i * step_size) % num_categories
        group_categories = []
        
        # 生成当前组的类别
        for j in range(group_size):
            category_id = (start_pos + j) % num_categories
            group_categories.append(category_id)
        
        # 去重但保持顺序
        seen = set()
        unique_categories = []
        for cat_id in group_categories:
            if cat_id not in seen:
                seen.add(cat_id)
                unique_categories.append(cat_id)
        
        groups.append(unique_categories)
    
        print(f"子集 {i}: 类别索引 {group_categories}，范围 {start_pos}-{start_pos + group_size - 1}")
    return groups

# 计算子集 subset classname
def get_cifar100_subset_class(overlap: float=0.2, model_num: int=3, num_classes=100) -> Tuple[int, int]:
    """
    计算子集的窗口大小和步长
    Args:       
    """
    return create_category_groups(num_categories=num_classes, num_groups=model_num, overlap_ratio=overlap)

    # overlap = 0.2
    # window_size = int(num_classes / model_num)
    # step = int(window_size * (1 - overlap))
    # if step < 1:
    #     step = 1
    # subsets = []
    # for i in range(model_num):
    #     if model_num == 1:
    #         start = 0
    #         end = num_classes
    #     elif i < model_num - 1:
    #         start = i * step
    #         end = start + window_size
    #     else:
    #         end = num_classes
    #         start = end - window_size
    #     class_list = list(range(start, end))
    #     subsets.append(class_list)
    #     print(f"子集 {i}: 类别索引 {class_list}，范围 {start}-{end}")
    # return subsets

# 使用示例
def create_cifar100_subset_dataloaders(
    class_list: List[int],
    batch_size: int = 32,
    num_workers: int = 4,
    root: str = './data'
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    创建CIFAR-100子集的训练和测试数据加载器
    
    Args:
        class_list: 要包含的类别列表
        batch_size: 批次大小
        num_workers: 工作进程数
        root: 数据根目录
    
    Returns:
        train_loader, test_loader
    """
    
    # 定义数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    # 创建训练和测试数据集
    train_dataset = CIFAR100Subset(
        root=root,
        train=True,
        class_list=class_list,
        transform=transform_train,
        download=True
    )
    
    test_dataset = CIFAR100Subset(
        root=root,
        train=False,
        class_list=class_list,
        transform=transform_test,
        download=True
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


# 使用示例
if __name__ == "__main__":
    # 示例1: 只包含前10个类别
    selected_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    train_loader, test_loader = create_cifar100_subset_dataloaders(
        class_list=selected_classes,
        batch_size=64
    )
    
    # 查看数据集信息
    print(f"训练集批次数: {len(train_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    
    # 获取一个批次的数据
    for images, labels in train_loader:
        print(f"批次图像形状: {images.shape}")
        print(f"批次标签: {labels[:10]}")  # 显示前10个标签
        break
    
    # 查看类别分布
    train_dataset = train_loader.dataset
    distribution = train_dataset.get_class_distribution()
    print(f"训练集类别分布: {distribution}")
    
    # 查看筛选后的类别名称
    filtered_names = train_dataset.get_filtered_class_names()
    print(f"筛选后的类别名称: {filtered_names}")
    
    # 示例2: 随机选择一些类别
    import random
    random_classes = random.sample(range(100), 20)  # 随机选择20个类别
    print(f"\n随机选择的类别: {sorted(random_classes)}")
    
    random_train_loader, random_test_loader = create_cifar100_subset_dataloaders(
        class_list=random_classes,
        batch_size=32
    )
    
    print(f"随机子集训练集批次数: {len(random_train_loader)}")
    print(f"随机子集测试集批次数: {len(random_test_loader)}")
