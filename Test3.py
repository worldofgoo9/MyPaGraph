# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 16:11:20 2021

@author: worldofgoo9
"""
# MyPaGraph Test
import os
import argparse
import sys
import MyPaGraph as pg
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import RedditDataset
from dgl.data import register_data_args
import time
import torch.multiprocessing as mp
import networkx as nx
import matplotlib.pyplot as plt
#from utils import thread_wrapped_func
from torch.nn.parallel import DistributedDataParallel
from multiprocessing import freeze_support
import tqdm
from utils import thread_wrapped_func

data = RedditDataset()
g = data[0]
TV,PV=pg.Init.DivideGraph(g=g,num=2,hop=1)

st = pg.Storage(g=g,data=g.ndata,gpu='cuda:0',cache_rate = 0.1,nodes=PV[0])
    
qn1 = torch.arange(140,5,-1)
qn2 = torch.arange(1400,2708)

r1 = st.Query(fname='label',nodes=TV[0],print_time=True)
#r2 = st.Query(fname='label',nodes=qn2)
  



