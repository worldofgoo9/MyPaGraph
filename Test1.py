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
import multiprocessing as mpr


class MyGraphConv(nn.Module):
    def __init__(self, in_feats, out_feats, activation = F.relu):
        super(MyGraphConv,self).__init__()
        self.W = nn.Linear(in_feats * 2, out_feats)
        self.activation = activation
        
    def forward(self, block, h):
        # with g.local_scope():
        with block.local_scope():
            # g.ndata['h'] = h
            h_src = h
            h_dst = h[:block.number_of_dst_nodes()]
            block.srcdata['h'] = h_src
            block.dstdata['h'] = h_dst

            # g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
            block.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))

            # return self.W(torch.cat([g.ndata['h'], g.ndata['h_neigh']], 1))
            return self.activation(self.W(torch.cat(
                [block.dstdata['h'], block.dstdata['h_neigh']], 1)))


class MyGCN(nn.Module):
    #This GCN model is using sampling method
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation=F.relu,
                 dropout=0.5,
                 g = None
                 ):
        super(MyGCN, self).__init__()
        #self.g = g
        assert g is None,"Remind that sampling method do not need Graph g"
        
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(MyGraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(MyGraphConv(n_hidden, n_hidden, activation=activation))
            
        # output layer
        self.layers.append(MyGraphConv(n_hidden, n_classes))       
        #self.dropout = nn.Dropout(p=dropout)

    def forward(self, blocks, h):
        assert len(blocks)==len(self.layers),\
        "Numbers of blocks and layers should be equal"
        assert blocks[0].number_of_src_nodes()==len(h),\
        "Number of src nodes should be equal to number of features(h)"
        #h = features
        
        #forward:
        for i, layer in enumerate(self.layers):
            '''
            if i != 0:
                h = self.dropout(h)
                #if use dropout
            '''            
            h = layer(blocks[i], h)
        return h        
        
def train(procid,args):
    # load and preprocess dataset
    assert procid >= 0
    os.environ['MASTER_ADDR'] = '127.0.0.1'              
    os.environ['MASTER_PORT'] = '12345'
    
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    elif args.dataset == 'reddit':
        data = RedditDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    g = data[0]
    
        
    #data = args.data
    #g = args.data[0]
    #g.create_formats_()
    print("New Proc! ",procid)
    #return g
    device = torch.device(args.devices_name_list[procid])
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
    world_size = args.ngpus
    torch.distributed.init_process_group(backend="nccl",
                                         init_method=dist_init_method,
                                          world_size = world_size,
                                          rank = procid)
    #torch.cuda.set_device(device)
#st = pg.Storage(g,[device],[args.PV_list[procid]],[args.TV_list[procid]])
    

    # use pagraph    
    st = pg.Storage(g=g,data=g.ndata,cache_rate=args.cache_rate,
                    nodes=args.PV_list[procid],gpu=args.devices_name_list[procid],cpu='cpu')
    if(True):
        features = g.ndata.pop('feat')
        labels = g.ndata.pop('label')
        train_mask = g.ndata.pop('train_mask')
        val_mask = g.ndata.pop('val_mask')
        test_mask = g.ndata.pop('test_mask')
        in_feats = features.shape[1]
        n_classes = data.num_labels
        n_edges = data.graph.number_of_edges()
        
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.int().sum().item(),
              val_mask.int().sum().item(),
              test_mask.int().sum().item()))
    
    del features    #release memory 

    # add self loop
    '''
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

    '''
    # create GCN model
    model = MyGCN(
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout,
                
                )
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids = [device], output_device = device)
    
    # set sampler
    fanouts=[]
    for i in range(args.n_layers + 1):
        fanouts.append(args.neighbor_number)
        '''
        example: fanout=[2,2,2,2] or [3,3,3] ...
        '''
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts) 
    train_nids = args.TV_list[procid]
    dataloader = dgl.dataloading.NodeDataLoader(    
    g, train_nids, sampler,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=0)
    
    # set loss function
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = args.lr)

    # initialize graph
    dur = []
    
        
    # Sync
    if(args.ngpus > 1):
        torch.distributed.barrier()   
        
    #Start trainning
    model.train()
    
    for epoch in range(args.n_epochs):
        # time record
        #if epoch >= 3:
        tS=[0.0,0.0,0.0,0.0,0.0,0.0]
        t0 = time.time()
        
        # forward

        #Loss=torch.tensor([0.0],device=device,required_grad=False)
        
        for count,(in_nodes,out_nodes,blocks) in enumerate(dataloader):
            
            t1=time.time()
            blocks=[b.to(device) for b in blocks]
            
            t2=time.time()
            feat_in = st.Query(0,in_nodes,'feat')
            labels_out = st.Query(0,out_nodes,'label')
            

            t3=time.time()
            # forward
            feat_out = model(blocks,feat_in)
            t4=time.time()
            
            loss = loss_fcn(feat_out,labels_out)
            #Loss=Loss+loss.detach()
            t5=time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t6=time.time()
            
            tS[1]=tS[1]+t2-t1
            tS[2]=tS[2]+t3-t2
            tS[3]=tS[3]+t4-t3
            tS[4]=tS[4]+t5-t4
            tS[5]=tS[5]+t6-t5
        
            
        
        tE=time.time()
        #logits = model(features)
        #loss = loss_fcn(logits[train_mask], labels[train_mask])
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()

        #if epoch >= 3:
        dur.append(time.time() - t0)

        acc = 0.0 #evaluate(model, features, labels, val_mask)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, n_edges / np.mean(dur) / 1000))
        #for i in range(1,6):
        print(tS[1:],'\nTotal:',tE-t0," s ")
        
        
    #Finish trainning
    
    # Sync
    if(args.ngpus > 1):
        torch.distributed.barrier()
    model.eval()

    print("____________________________")
    #acc = evaluate(model, features, labels, test_mask)
    #print("Test accuracy {:.2%}".format(acc))
    



if __name__ == '__main__':
    
    freeze_support()

    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--data", type=str, default="pubmed",
                        help="Dataset")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--ngpus", type=int, default=2,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=4,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--hop", type=int, default=1,
                        help="number hop")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="number of batch size")
    parser.add_argument("--neighbor-number", type=int, default=5,
                        help="number of neighbor")
    parser.add_argument("--cache-rate", type=float, default=0.5,
                        help="Cache rate")

    args = parser.parse_args()
    args.dataset=args.data
    args.devices_name_list = []
    for i in range(args.ngpus):
        args.devices_name_list.append('cuda:'+str(i))
    args.devices_name_list = ['cuda:0','cuda:0']
    
    print("args: \n",args)
    
    os.environ['MASTER_ADDR'] = '127.0.0.1'              #
    os.environ['MASTER_PORT'] = '12345'
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    elif args.dataset == 'reddit':
        data = RedditDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    g = data[0]


    args.TV_list,args.PV_list=pg.Init.DivideGraph(g,args.ngpus,args.hop)
    
    del data
    del g
    # release memory
       
    mp.spawn(train,nprocs = args.ngpus,args = (args,))
    
    print("Exit!")
    
