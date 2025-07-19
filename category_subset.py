def create_category_groups(num_categories, num_groups, overlap_ratio):
    """
    根据overlap比例分配类别到不同组，相邻组重复指定比例的类别
    
    Args:
        num_categories (int): 类别总数
        num_groups (int): 分组数量
        overlap_ratio (float): 重复比例 (0-1之间)
                              表示相邻两组重复类别数量占总类别数的比例
    
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
    
    # 计算每组需要的新类别数量（不包括与前一组重复的部分）
    new_categories_per_group = (num_categories - overlap_count * num_groups) // num_groups
    
    # 如果新类别数量为负数或太小，说明重复太多，需要调整
    if new_categories_per_group <= 0:
        # 重复太多的情况，每组至少要有一些新类别
        new_categories_per_group = max(1, num_categories // num_groups)
        # 重新计算实际的overlap
        overlap_count = min(overlap_count, num_categories - new_categories_per_group)
    
    # 计算每组的总大小
    group_size = new_categories_per_group + overlap_count
    
    groups = []
    category_pool = list(range(num_categories))
    
    # 构建循环序列，如果类别数不够，就重复序列
    # 计算需要的总长度
    total_needed = num_groups * new_categories_per_group + overlap_count
    
    # 如果需要的长度超过类别数，创建循环序列
    if total_needed > num_categories:
        cycles_needed = (total_needed + num_categories - 1) // num_categories
        extended_categories = (category_pool * cycles_needed)[:total_needed]
    else:
        extended_categories = category_pool
    
    for i in range(num_groups):
        group_categories = []
        
        # 添加与前一组重复的类别
        if i > 0 and overlap_count > 0:
            # 取前一组的最后overlap_count个类别
            prev_group = groups[i-1]
            overlap_categories = prev_group[-overlap_count:]
            group_categories.extend(overlap_categories)
        
        # 添加新的类别
        start_idx = i * new_categories_per_group
        if i == 0:
            # 第一组不需要考虑重复，直接取前group_size个
            new_start = 0
        else:
            # 后续组需要跳过已使用的类别
            new_start = overlap_count + i * new_categories_per_group
        
        # 确保不超出扩展序列的范围
        for j in range(new_categories_per_group):
            if new_start + j < len(extended_categories):
                category_id = extended_categories[new_start + j]
                if category_id not in group_categories:
                    group_categories.append(category_id)
        
        # 如果第一组需要补充类别以达到目标大小
        if i == 0:
            while len(group_categories) < group_size and len(group_categories) < num_categories:
                for cat_id in extended_categories:
                    if cat_id not in group_categories:
                        group_categories.append(cat_id)
                        break
                else:
                    break
        
        groups.append(group_categories)
    
    return groups


def create_category_groups_v2(num_categories, num_groups, overlap_ratio):
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
    
    return groups


def analyze_overlap(groups, num_categories):
    """
    分析组之间的重复情况
    
    Args:
        groups (list): 组列表
        num_categories (int): 总类别数
    
    Returns:
        dict: 重复分析结果
    """
    if len(groups) <= 1:
        return {"adjacent_overlaps": [], "total_unique_categories": len(groups[0]) if groups else 0}
    
    adjacent_overlaps = []
    all_categories = set()
    
    for i in range(len(groups)):
        # 计算与下一组的重复（包括最后一组与第一组）
        next_i = (i + 1) % len(groups)
        overlap = set(groups[i]) & set(groups[next_i])
        adjacent_overlaps.append({
            "groups": f"{i+1}-{next_i+1}",
            "overlap_categories": sorted(overlap),
            "overlap_count": len(overlap),
            "overlap_ratio_of_total": len(overlap) / num_categories,
            "overlap_ratio_of_group": len(overlap) / len(groups[i]) if groups[i] else 0
        })
    
    for group in groups:
        all_categories.update(group)
    
    return {
        "adjacent_overlaps": adjacent_overlaps,
        "total_unique_categories": len(all_categories),
        "group_sizes": [len(group) for group in groups]
    }


def test_category_grouping():
    """测试函数"""
    test_cases = [
        (10, 3, 0.2),   # 20%重复 = 2个类别重复
        (10, 4, 0.3),   # 30%重复 = 3个类别重复
        (8, 5, 0.25),   # 25%重复 = 2个类别重复（需要循环）
        (100, 3, 0.5),    # 50%重复 = 3个类别重复（需要循环）
        (12, 3, 0.1),   # 10%重复 = 1个类别重复
    ]
    
    for num_categories, num_groups, overlap_ratio in test_cases:
        print(f"\n=== 测试: {num_categories}个类别，{num_groups}个组，overlap={overlap_ratio} ===")
        print(f"预期重复类别数: {int(num_categories * overlap_ratio)}")
        
        # 测试两个版本
        groups1 = create_category_groups(num_categories, num_groups, overlap_ratio)
        groups2 = create_category_groups_v2(num_categories, num_groups, overlap_ratio)
        
        print("\n版本1结果:")
        for i, group in enumerate(groups1):
            print(f"组 {i+1}: {group}")
        
        print("\n版本2结果:")
        for i, group in enumerate(groups2):
            print(f"组 {i+1}: {group}")
        
        # 分析版本2的重复情况
        analysis = analyze_overlap(groups2, num_categories)
        print(f"\n分析结果:")
        print(f"组大小: {analysis['group_sizes']}")
        print(f"总共覆盖类别数: {analysis['total_unique_categories']}")
        
        print("相邻组重复情况:")
        for overlap_info in analysis['adjacent_overlaps']:
            print(f"  组{overlap_info['groups']}: {overlap_info['overlap_count']}个重复类别 "
                  f"({overlap_info['overlap_ratio_of_total']:.1%} of total) "
                  f"{overlap_info['overlap_categories']}")


if __name__ == "__main__":
    test_category_grouping()
