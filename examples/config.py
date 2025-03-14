# config.py
import os
import argparse
from yacs.config import CfgNode as CN

cfg = CN()

cfg.n_mlp_layers = 2
cfg.mlp_hidden_dims = 40
cfg.mlp_use_bn = True
cfg.mlp_activation = 'relu'
cfg.mlp_dropout_prob = 0
cfg.n_phi_layers = 7
cfg.hidden_phi_layers = 40
cfg.pe_dims = 40
cfg.BASIS = False
cfg.RAND_k = 10
cfg.RAND_mlp_nlayers = 1
cfg.RAND_mlp_hid = 40
cfg.RAND_act = 'relu'
cfg.RAND_mlp_out = 40
cfg.node_emb_dims = 128
cfg.num_samples = 100
cfg.PE1 = False
cfg.splits = 4
cfg.smallest = False


def load_config_from_file(config_path):
    if config_path and os.path.exists(config_path):
        cfg.merge_from_file(config_path)

def merge_config(path):
    if path is not None:
        load_config_from_file(path)
    return cfg

