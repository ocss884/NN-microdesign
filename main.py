"""
main.py
"""

import torch
from torch import nn, optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
from modules.utils import load_dataset, ODFDataset, seed_worker, timer, plot_loss, exp_weight_algo
from modules.nn import NN
from tqdm import tqdm
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STIFF_MTX = np.genfromtxt('./data/stiffness.csv', delimiter=',')
VOL = np.genfromtxt('./data/volumefraction.csv', delimiter=',')
lamda = ((np.ones((6,6))+np.eye(6))/2).reshape(-1)
# WMSE weight
W = torch.Tensor(lamda@STIFF_MTX/(min(lamda@STIFF_MTX))).to(device)

g = torch.Generator()
g.manual_seed(0)

def train():
    LOSS = np.empty(0)
    opt = [
        (optim.Adam, {"lr": 5e-5}, 100),
        (optim.SGD, {"lr": 1e-4, "momentum":.9}, 200),
        (optim.Adam, {"lr": 2e-5}, 1000),
        (optim.Adam, {"lr": 1e-4}, 500),
        (optim.Adam, {"lr": 2e-5}, 1000),
        (optim.Adam, {"lr": 1e-4}, 500),
        (optim.Adam, {"lr": 2e-5}, 1000),
        ]
    for i in range(len(opt)):
        optimizer = opt[i][0](model.parameters(), **opt[i][1])
        for epoch in tqdm(range(opt[i][2])):
            training_loss = 0
            for p, q in train_loader:
                p = p.to(device)
                q = q.to(device)
                optimizer.zero_grad()
                pred_q = model(p)
                loss = loss_fn(q*W, pred_q*W)
                training_loss += loss
                loss.backward()
                optimizer.step()
            LOSS = np.append(LOSS, training_loss.detach().cpu().numpy())
            print(f"\n{epoch=}: WMSE LOSS={training_loss:.4f}")
    plot_loss(LOSS)

def test():
    with torch.no_grad():
        for p, q in test_loader:
            p = p.to(device)
            q = q.to(device)
            pred_q = model(p)
            test_loss = loss_fn(q*W, pred_q*W)
            stiff_err = np.mean(abs(pred_q.cpu().numpy()@(lamda@STIFF_MTX)-q.cpu().numpy()@(lamda@STIFF_MTX)), axis=0)
    if device != torch.device("cpu"):
        print(f"Model inference time {timer(model, device):.4f} seconds on {torch.cuda.get_device_name(device)}")
    print(f"Test WMSEloss: {test_loss:.4f}\n"
          f"1-step Stiffness Error on all deformations: {stiff_err}\n")

if __name__ == '__main__':
    # load dataset
    X, Y = load_dataset()
    dataset = ODFDataset(X, Y)
    
    # set seed
    torch.manual_seed(3407)
    
    # Create model, train/test dataset, loss function
    model = NN(VOL, num_modes=31).to(device)
    loss_fn = nn.MSELoss()
    train_dst, test_dst = random_split(dataset, [4000, 1000])
    train_loader = DataLoader(train_dst, 128, shuffle=True)
    test_loader = DataLoader(test_dst, 1000, worker_init_fn=seed_worker, generator=g, shuffle=False)

    train()
    test()