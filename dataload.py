import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle

class FlowDataset(object):
    def __init__(self):
        self.X = np.load('flow_data.npy')
        with open('labels.pkl', 'rb') as f:
            self.y = pickle.load(f)

    def __getitem__(self, index):
        X = torch.Tensor(self.X[index]).type(torch.uint8).to('cuda')
        y = self.y[index]

        return X, y

    def __len__(self):
        return self.X.shape[0]

if __name__ == '__main__':
    dataset = FlowDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for batch_id, sample in enumerate(dataloader):
        X, y = sample
        print(X.shape, y.shape)
        print(X, y)
        break
