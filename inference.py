from model import HAM

from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn.functional as F

import pickle
import colored_glog as log


def read_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--flow',
        required=True,
        type=str,
        help='Filepath to network flow data'
    )

    return parser.parse_args()


def load_model():
    ham = HAM().to('cuda')
    ham.load_state_dict(torch.load('ham.pt'))

    return ham.eval()

if __name__ == '__main__':
    args = read_args()
    ham = load_model()

    with open('label_map.pkl', 'rb') as f:
        label_map = pickle.load(f)

    data = torch.Tensor(np.load(args.flow)).type(torch.long).unsqueeze(0).to('cuda')
    prediction = ham(data)

    scores = F.softmax(prediction, dim = 1).squeeze()

    item_id = prediction.argmax().item()

    predicted_item_name = label_map[item_id]
    confidence = scores[item_id]

    log.info('Predicted => {} (Confidence {:.2f}%)'.format(predicted_item_name, confidence * 100))
