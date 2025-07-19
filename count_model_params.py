import torch
from models import model_dict

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def main():
    num_classes = 100  # 以cifar100为例
    results = []
    for name, builder in model_dict.items():
        try:
            # 部分模型可能不支持num_classes参数
            try:
                model = builder(num_classes=num_classes)
            except TypeError:
                model = builder()
            params = count_params(model)
            results.append((name, params))
        except Exception as e:
            print(f"模型 {name} 加载失败: {e}")

    # 按参数量排序
    results.sort(key=lambda x: x[1])

    print(f"{'Model':<20} {'Params (M)':>12}")
    print("-" * 32)
    for name, params in results:
        print(f"{name:<20} {params/1e6:>10.2f}M")

if __name__ == "__main__":
    main() 
