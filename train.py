from utils import read_config
from model import HAM
from dataload import FlowDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import colored_glog as log

if __name__ == '__main__':
    config = read_config()

    ham = HAM().to('cuda')
    flow_dataset = FlowDataset()
    flow_dataloader = DataLoader(
        flow_dataset,
        batch_size=eval(config['train']['BatchSize']),
        shuffle=eval(config['train']['Shuffle'])
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ham.parameters(), lr=1e-3, weight_decay=1e-5)

    for epoch in range(eval(config['train']['NumEpochs'])):
        log.info('Running Epoch #{}'.format(epoch + 1))
        total_loss = 0
        with tqdm(total = len(flow_dataset) // eval(config['train']['BatchSize'])) as pbar:
            for batch_id, sample in enumerate(flow_dataloader):
                pbar.update(1)
                optimizer.zero_grad()
                X, y = sample
                prediction = ham(X.type(torch.long))

                loss = criterion(prediction, y.to('cuda'))
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

        log.info('Epoch #{} ==> {:.4f}'.format(epoch+1, total_loss))

    torch.save(ham.state_dict(), 'ham.pt')
