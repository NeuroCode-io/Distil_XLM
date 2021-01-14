import random
import torch
import os
import numpy as np

TRAIN_TEST_RATIO = 0.2
ROOT_DATA_PATH = '../data/SST-2'
TRAIN_FILE = os.path.join(ROOT_DATA_PATH, 'train.tsv')
TEST_FILE = os.path.join(ROOT_DATA_PATH, 'test.tsv')
DEV_FILE = os.path.join(ROOT_DATA_PATH, 'dev.tsv')

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

xlm_settings = {
    'max_seq_length': 128,
    'num_train_epochs': 4,
    'train_batch_size': 16,
    'eval_batch_size': 16,
    'learning_rate': 1e-5,
    'adam_epsilon': 1e-8,
    'test_size': TRAIN_TEST_RATIO,
    'tb_suffix': 'xlm'
}

distillation_settings = {
    'max_seq_length': 128,
    'num_train_epochs': 30,
    'train_batch_size': 32,
    'eval_batch_size': 32,
    'test_size': TRAIN_TEST_RATIO,
    'tb_suffix': 'd_lstm'
}

def batch_to_inputs(batch):
    inputs = {
        "input_ids": batch[0],
        "attention_mask": batch[1],
        "token_type_ids": batch[2],
        "labels": batch[3],
    }

    return inputs