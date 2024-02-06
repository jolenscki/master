import os
from os import path as osp
import sys
import time
from tqdm import tqdm
import pickle
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union
from typing import Dict
from typing import List
from itertools import cycle
import matplotlib.pyplot as plt
import networkx as nx

import numpy as np
# pytorch imports
import torch
from torch import Tensor, nn, cuda
from torch.nn import Conv3d
import torch.nn.functional as F
from torch.utils.data import random_split
# pytorch geometric imports
import torch_geometric
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
# pytorch geometric temporal imports
from torch_geometric_temporal.nn.recurrent import GConvLSTM

# lightning imports
from lightning.pytorch.utilities.combined_loader import CombinedLoader

from util.h5_util import load_h5_file, write_data_to_h5
from util.data_range import generate_date_range, weekday_parser, date_parser
from util.graph_utils import reconstruct_graph, poor_man_graph_visualization
from util.graph_cell_structure import (offset_map, volume_channel_indices,
                                       speed_channel_indices)

from util.utils import align_datasets

from data.dataset.ImageDataset import ImageDataset
from data.dataset.GraphDataset import GraphDataset

from model.transform import collapse_channels
from model.MMD import MMDLoss, RBF

class ExtractionNet(nn.Module):
    '''
    First part of the network, where ST features should be
    extracted from the graphs
    
    Parameters
    ----------
    num_channels: int
    hidden_size: int
        power of 2
    window_length: list[int]
        list of the extension of the input and output windows length;
        basically the number of snapshots we have and how many we want to 
        predict
    num_layers: int
        number of ConvLSTM cells to be stacked
    '''
    def __init__(self,
                 num_channels: int,
                 hidden_size: int,
                 window_length: List[int],
                 num_layers: Optional[int]=3):
        super(ExtractionNet, self).__init__()
        # first we define internal variables based on class input
        self.num_channels     = num_channels
        self.hidden_size = hidden_size
        self.input_snapshots, self.output_snapshots = window_length
        self.num_layers = num_layers
        
        # define network
        self.CLSTM = GConvLSTM(self.num_channels, self.hidden_size, K=3)
    
    @torch.autocast(device_type="cuda")
    def forward(self, x, edge_index, h, c):
        # first of all, reshape x so instead of [nodes, 24] we have 
        # [nodes, 2, 12], as we have 12 snapshots of 2 channels
        xsize = x.shape[0]
        x_ = x.reshape(xsize, self.num_channels, self.input_snapshots)
        H, C = [], []
        
        for t in range(self.input_snapshots):
            x = x_[..., t]
            # third keyword is edge_attr/weight, but we don't use it
            for _ in range(self.num_layers):
                h, c = self.CLSTM(x, edge_index, None, h, c)
                h, c = h.detach(), c.detach()
            
            H.append(h)
            C.append(c)
        H = torch.stack(H)
        C = torch.stack(C)
        
        return H, C
    
class EmbeddingNet(nn.Module):
    '''
    Second part of the network, where the features are embedded in a common
    space
    
    Parameters
    ----------
    num_channels: int
    num_layers: int
    tgt: str
    
    hidden_size: Optional[list[int]]
    '''
    def __init__(self, 
                 num_channels: int,
                 num_layers: int,
                 tgt: str,
                 hidden_size=[16, 32]):
        super(EmbeddingNet, self).__init__()
        self.num_channels = num_channels
        self.num_layers   = num_layers
        self.tgt          = tgt
        self.hidden_size  = hidden_size
        assert num_layers == len(hidden_size), "hidden_size parameter seems wrong"
        
        # MMD Loss 'model'
        self.mmd = MMDLoss(batch_size=2**10)
        self.cells = nn.ModuleList()
        for i in range(self.num_layers):
            inc = self.num_channels if i == 0 else self.hidden_size[i-1]
            out = self.hidden_size[i]
            self.cells.append(
                nn.Conv2d(
                    in_channels=inc,
                    out_channels=out,
                    kernel_size=(3, 3),
                    padding=(1, 1),
                )
            )
        self.mmd_loss = 0
    
    def forward(self, h_cities):
        '''
        Parameters
        ----------
        h_cities: dict[torch.Tensor]
        '''
        # self.embeds = {city: [] for city in h_cities.keys()}
        src_cities = [city for city in h_cities.keys() if city not in self.tgt]
        # first, iterate over the layer
        for layer in range(self.num_layers):
            # iterate over the cities then
            for city, h in h_cities.items():
                new_h = self.cells[layer](h)
                # self.embds[city].append(new_h)
                h_cities[city] = new_h
            # after calculating all new_h, or Conv(h) for every city, we
            # calculate the mmd loss between the tgt city and all others
            h_tgt = h_cities[self.tgt]
            for src_city in src_cities:
                h_src = h_cities[src_city]
                self.mmd_loss += self.mmd(h_src, h_tgt)
        
        # reshape h_cities
        for k, v in h_cities.items():
            h_cities[k] = torch.movedim(v, 1, 0)
        
        return h_cities, self.mmd_loss