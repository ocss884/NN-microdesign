import numpy as np
from numpy.random import multinomial
import torch
from torch.utils.data import Dataset

def load_dataset():
    """
    Load dataset from files. Later split into train and test set.
    """
    init_ODF = np.load("./data/init_ODF.npy")

    X = torch.Tensor(init_ODF[:5000])
    Y = np.zeros((5000, 31, 145))
    for i in range(31):
        Y[:, i, :] = np.load(f"./data/dataset/SIMU_{i:02}.npy")
    Y = torch.Tensor(Y)
    return X[:,:76], Y[:,:,:76]

class ODFDataset(Dataset):
    """
    Dataset class for ODF data.
    Args:
        x (torch.Tensor): Initial ODFs.
        y (torch.Tensor): Simulated all possible ODFs.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        sample = (self.x[idx], self.y[idx])
        return sample

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

# Searing algorithm
def exp_weight_algo(model, init_ODF):
    """
    Exponential weights Searing algorithm to find the optimal path.
    Args:
        model (nn.Module): trained Neural network model.
        init_ODF (np.ndararry): initial ODFs.
    """
    HASH = {}
    with torch.no_grad():
        for _ in range(1000):
            path = []
            ipt = init_ODF
            for i in range(10):
                ipt = torch.Tensor(ipt).cuda()
                opt = model(ipt).cpu().numpy()
                stf = ((opt-ipt.cpu().numpy())@(lamda@Q))[0,:]
                if (stf>0).sum()==0:
                    continue
                weights = np.exp(5*stf)
                weights = weights/weights.sum()
                idx = np.where(multinomial(1, weights) == 1)[0][0]
            
                ipt = opt[:,idx,:]
                path.append(idx)
            f_stf = opt[0,idx,:]@(lamda@Q)
            HASH[f_stf] = path
    return HASH

def timer(model, device):
    """
    Timer to measure the inference time.
    """

    dummy_input = torch.randn(1, 76, dtype=torch.float).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1000
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    return mean_syn

def plot_loss(loss):
    """
    Plot loss function.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(dpi=600)
    num_iters = len(loss)
    loss = np.log10(loss)
    plt.plot(range(num_iters), loss)
    plt.xlabel("iter. (1e3)")
    plt.ylabel("WMSEloss (log$_{10}$)")
    plt.savefig("loss.png")