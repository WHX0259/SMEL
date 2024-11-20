import pandas as pd
import json
import os

# 1. 读取标签文件
label_path = '/data/huixuan/data/data_chi/label.csv'
labels = pd.read_csv(label_path)

# 将ID和标签存储在字典中
label_dict = {row['ID'].split('.')[0]: row['label'] for _, row in labels.iterrows()}

# 2. 读取十折交叉验证文件
folds_path = '/data/huixuan/data/data_chi/TRG_patient_folds.json'
with open(folds_path, 'r') as f:
    folds = json.load(f)

# 3. 统计每个Fold中每个类别的数量
fold_counts = {}

for fold_name, images in folds.items():
    fold_counts[fold_name] = {0: 0, 1: 0}  # 初始化每个类别的计数
    for img in images:
        # 获取对应的ID
        id_key = img.split('_')[0]
        # 如果在标签字典中找到对应的标签
        if id_key in label_dict:
            label = label_dict[id_key]
            fold_counts[fold_name][label] += 1  # 统计类别数量

# 4. 保存结果到指定路径
output_path = '/data/wuhuixuan/data/fold_counts.csv'
fold_counts_df = pd.DataFrame(fold_counts).T  # 转置，使得Fold名称成为行
fold_counts_df.columns = ['Count_0', 'Count_1']  # 重命名列
fold_counts_df.to_csv(output_path, index=True)  # 保存到CSV文件

print(f"统计结果已保存到 {output_path}")
