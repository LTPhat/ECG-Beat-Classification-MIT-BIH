import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ECGDataset(Dataset):
    def __init__(self, data_path, labels_path, lead=0):
        self.data = np.load(data_path)  # shape: (N, 250)
        self.labels = np.load(labels_path)  # shape: (N,)
        self.lead = lead  # choose lead 0 or 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        ecg_sample = self.data[idx, :].astype(np.float32)  # shape: (250,)
        label = torch.tensor(self.labels[idx])
        return ecg_sample, label