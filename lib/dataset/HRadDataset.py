import os
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch
import json
# This class `TumorDataset` is a custom PyTorch dataset for loading medical imaging data with
# associated metadata and labels for tumor classification tasks.
class TumorDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir='/data/wuhuixuan/code/Self_Distill_MoE/data/square_cropping32/image', 
                 mask_dir='/data/wuhuixuan/code/Self_Distill_MoE/data/square_cropping32/mask', 
                 tumor_dir='/data/wuhuixuan/code/Self_Distill_MoE/data/square_cropping32/tumor', 
                 metadata_path='/data/wuhuixuan/code/Self_Distill_MoE/data/square_cropping32/metadata.csv', 
                 csv_path = '/data/huixuan/data/data_chi/label.csv',
                 fold_json='/data/wuhuixuan/data/data_chi_Mediastinum/TRG_patient_folds.json', fold=0,mode = 'train', transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.tumor_dir = tumor_dir
        self.metadata = pd.read_csv(metadata_path)
        self.image_list = []
        with open(fold_json, 'r') as file:
            data = json.load(file)
        if mode == 'train':
            for i in range(10):
                if i != fold:
                    for case_id in data['Fold ' + str(i + 1)]:
                        self.image_list.append(case_id)
        elif mode == 'val':
            for case_id in data['Fold ' + str(fold + 1)]:
                self.image_list.append(case_id)
        if transform==None:
            self.transform = transforms.Compose([
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomVerticalFlip(p=0.5),
                            transforms.ToTensor(),  # 转换为Tensor
                        ])
        else:
            self.transform = transform
        df = pd.read_csv(csv_path)
        self.TRG_dict = dict(zip(df['ID'], df['label']))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # 获取ID和元数据
        if self.image_list[index].endswith('.png'):
            image_name = self.image_list[index].split('.')[0].split('_')[0]
        elif self.image_list[index].endswith('.nii.gz'):
            image_name = self.image_list[index]
        id_ = int(image_name)
        row = self.metadata.loc[self.metadata['ID'] == id_]
        normalized_size = row['Normalized_Size'].values[0]
        normalized_age = row['Normalized_Age'].values[0]

        # 构建文件路径
        image_path = os.path.join(self.image_dir, f"{id_}_image.png")
        mask_path = os.path.join(self.mask_dir, f"{id_}_mask.png")
        tumor_path = os.path.join(self.tumor_dir, f"{id_}_tumor.png")

        # 读取图像
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")  # 假设mask是单通道
        tumor = Image.open(tumor_path).convert("L")
        # 合并通道
        combined_image = Image.merge("RGB", (image, mask, tumor))

        # 数据增强
        if self.transform:
            combined_image = self.transform(combined_image)
        # 将PIL图像转换为PyTorch张量
        # combined_tensor = transforms.ToTensor()(combined_image)

        # 读取元数据
        row = self.metadata[self.metadata['ID'] == id_]
        if row.empty:
            raise ValueError(f"ID {id_} not found in metadata.")
        
        normalized_size = row['Normalized_Size'].values[0]
        normalized_age = row['Normalized_Age'].values[0]

        # 创建元数据张量
        metadata_tensor = torch.tensor([normalized_size, normalized_age], dtype=torch.float32)
        class_id = self.TRG_dict[image_name + '.nii.gz']
        return combined_image, metadata_tensor, class_id, id_

if __name__ == '__main__':
    transform = transforms.Compose([
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.ToTensor(),  # 转换为Tensor
                            ])
    # 实例化数据集
    image_dir = '/data/wuhuixuan/code/Self_Distill_MoE/data/square_cropping32/image'
    mask_dir = '/data/wuhuixuan/code/Self_Distill_MoE/data/square_cropping32/mask'
    tumor_dir = '/data/wuhuixuan/code/Self_Distill_MoE/data/square_cropping32/tumor'
    metadata_path = '/data/wuhuixuan/code/Self_Distill_MoE/data/square_cropping32/metadata.csv'

    dataset = TumorDataset(mode='val',fold=3)

    # 示例：加载数据
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)

    for combined, metadata,label,ID_ in data_loader:
        print(combined.shape)  # 输出形状 [1, 3, H, W]，其中H和W为图像尺寸
        print(metadata)  # 输出对应的Normalized_Size和Normalized_Age
        print(label)