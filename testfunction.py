
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from utils import get_data, get_data_dim, get_loader
from eval_method import bf_search
from tqdm import tqdm

from Model import MUTANT

class ExpConfig():
    dataset = "SWaT"
    val = 0.35  # the ratio of validation set
    max_train_size = None  # `None` means full train set
    train_start = 0

    max_test_size = None  # `None` means full test set
    test_start = 0

    input_dim = get_data_dim(dataset)
    batch_size = 120

    out_dim = 5   # the dimension of embedding
    window_length = 20
    hidden_size = 100  # the dimension of hidden layer in LSTM-based attention
    latent_size = 100  # the dimension of hidden layer in VAE
    N = 256

def main():
    config = ExpConfig()

    (train_data, _), (test_data, test_label) = \
        get_data(config.dataset, config.max_train_size, config.max_test_size, train_start=config.train_start,
                 test_start=config.test_start)

    n = int(test_data.shape[0] * config.val)
    test_data = test_data[:-n]
    test_label = test_label[:-n]

    val_data = test_data[-n:]
    val_label = test_label[-n:]

    print("test_data:", test_data.shape)
    print("val_data:", val_data.shape)

    train_data = train_data[np.arange(config.window_length)[None, :] + np.arange(train_data.shape[0] - config.window_length)[:, None]]
    val_data = val_data[np.arange(config.window_length)[None, :] + np.arange(val_data.shape[0] - config.window_length)[:, None]]
    test_data = test_data[np.arange(config.window_length)[None, :] + np.arange(test_data.shape[0] - config.window_length)[:, None]]

    num_val = int(val_data.shape[0]/config.batch_size)
    con_val = val_data.shape[0] % config.batch_size
    num_t = int(test_data.shape[0]/config.batch_size)
    con_t = test_data.shape[0] % config.batch_size

    train_loader = get_loader(train_data, batch_size=config.batch_size,
                              window_length=config.window_length, input_size=config.input_dim, shuffle=True)
    val_loader = get_loader(val_data, batch_size=config.batch_size,
                              window_length=config.window_length, input_size=config.input_dim, shuffle=True)
    test_loader = get_loader(test_data, batch_size=config.batch_size,
                              window_length=config.window_length, input_size=config.input_dim, shuffle=False)


    print("1111\n", num_val)
    print(len(val_loader))


if __name__ == '__main__':
    main()