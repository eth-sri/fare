import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from box import Box
from src.common.dataset import get_dataset


class _dataloader:
    
    def __init__(self, dataset, batch_size, scaling):
        """_summary_
        Args:
            dataset (str): select one from {adult, compas, health}
            batch_size (int): batch size
            scaling (bool): scaling input vectors for reconstruction
        """
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.scaling = scaling
        self.scaler = MinMaxScaler()
        
        data, meta = get_dataset(Box({'name': self.dataset, 'val_size': 0}))
        self.data = data
        self.meta = meta
        
    def _to_dataloader(self, data, batch_size):
        x = torch.tensor(data['train'][0]).float()
        y = torch.tensor(data['train'][2].reshape(-1,1)).float()
        s = torch.tensor(data['train'][1].reshape(-1,1)).float()
        
        TD = TensorDataset(x, y, s)
        sampler = RandomSampler(TD)
        train_dataloader = DataLoader(TD, sampler=sampler, batch_size=batch_size)
               
        return train_dataloader
    
    # train dataloader
    def train(self):
        return self._to_dataloader(self.data, self.batch_size)
    
    # val dataloader
    def val(self):
        x = torch.tensor(self.data['train'][0]).float()
        y = torch.tensor(self.data['train'][2].reshape(-1,1)).float()
        s = torch.tensor(self.data['train'][1].reshape(-1,1)).float()
        return x, y, s
    
    # test dataloader
    def test(self):
        x = torch.tensor(self.data['test'][0]).float()
        y = torch.tensor(self.data['test'][2].reshape(-1,1)).float()
        s = torch.tensor(self.data['test'][1].reshape(-1,1)).float()
        return x, y, s
    
