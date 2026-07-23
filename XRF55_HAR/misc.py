import argparse
import os
import random
from typing import List
import torch
import numpy as np

def random_num_select(inputs:List):
    modality_num = len(inputs)
    select_input = random.sample(inputs, random.randint(1, modality_num))
    other_input = set(inputs) - set(select_input)
    
    return select_input, other_input, inputs

def all_num_select(inputs: List[str]):
    return inputs, []
def extant_file(x):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x

def adjust_learning_rate(optimizer, i_iter, lr, num_steps, power):
    lr = lr_poly(lr, i_iter, num_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def collate_fn_padd(batch):
    '''
    Padds batch of variable length
    '''
    labels = []
    [labels.append(int(t[3])) for t in batch]
    labels = torch.LongTensor(labels)

    # mmwave
    mmwave_data = np.array([(t[2]) for t in batch])
    mmwave_data = torch.FloatTensor(mmwave_data)

    # wifi-csi
    wifi_data = np.array([(t[0]) for t in batch])
    wifi_data = torch.FloatTensor(wifi_data)

    # rfid
    rfid_data = np.array([(t[1]) for t in batch])
    rfid_data = torch.FloatTensor(rfid_data)

    return mmwave_data, wifi_data, rfid_data, labels