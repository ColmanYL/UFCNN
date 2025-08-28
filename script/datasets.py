from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from tqdm import tqdm
import csv
import pandas as pd
import torch

class MyDataset(Dataset):
    def __init__(self, root_dir: str, csv_file, num_points: int, in_memory=False, take: int = -1):
        super(MyDataset, self).__init__()  # 这里修改为 MyDataset
        self.root = root_dir
        self.filelist = csv_file
        self.in_memory = in_memory
        self.take = take
        self.data_frame = pd.read_csv(self.filelist)
        self.filenames, self.labels = self.load_filenames()

        if self.in_memory:
            print('Load files into memory from ' + self.filelist)
            self.samples = [self.read_file(os.path.join(self.root, f))
                            for f in tqdm(self.filenames, ncols=80, leave=False)] 

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # 直接使用 idx 而不是 idx-1
        data = np.load(self.filenames[idx])  # 加载 .npy 文件
        
        # 将 NumPy 数组转换为 PyTorch Tensor
        tensor_data = torch.from_numpy(data).float()
        
        # 获取标签并检查其类型
        label_value = self.labels[idx]
        label_value = float(label_value)  # 将标签转换为 float
        label = torch.tensor(label_value, dtype=torch.float32)  # 转换为 Tensor

        return tensor_data, label
    
    def load_filenames(self):
        filenames, labels = [], []
        with open(self.filelist) as csvfile:
            csv_reader = csv.reader(csvfile)
            header = next(csv_reader)
            for row in csv_reader:
                table_path = os.path.join(self.root, f"{row[0]}.pt_distances.npy")
                filenames.append(table_path)
                labels.append(row[1])

        num = len(filenames)
        if self.take > num or self.take < 1:
            self.take = num
        
        return filenames[:self.take], labels[:self.take]
