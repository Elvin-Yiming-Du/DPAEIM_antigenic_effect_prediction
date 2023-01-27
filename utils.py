import numpy as np
import torch
from torch.utils.data import TensorDataset, SubsetRandomSampler
random_seed = 42
def shuffle_dataset(dataset, batch_size):
    indices = [i for i in range(len(dataset))]

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    sampler = SubsetRandomSampler(indices)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=sampler, drop_last=True)
    return data_loader