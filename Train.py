# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 09:55:25 2021

@author: worldofgoo9
"""
import argparse
import sys
import dgl
import numpy as np
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import networkx as nx
import matplotlib.pyplot as plt
import time
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset,RedditDataset



#import MyPaGraph as pg

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
        
def main(args,mode=None):
    # load and preprocess dataset
    '''
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    '''
    data = RedditDataset()

    
    g = data[0]
    g.create_formats_()
    #return g
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    '''
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)
    '''
    if(mode==None):
        features = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        in_feats = features.shape[1]
        n_classes = data.num_labels
        n_edges = data.graph.number_of_edges()
    else:
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

    # add self loop
    '''
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    '''
    '''
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)
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
    model=model.to(device)
    '''
    if cuda:
        model.cuda()
    '''
    # set sampler
    fanouts=[]
    for i in range(args.n_layers + 1):
        fanouts.append(args.neighbor_number)
        '''
        example: fanout=[2,2,2,2] or [3,3,3] ...
        '''
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts) 
    train_nids=torch.arange(0,len(train_mask),1,dtype=torch.long)
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
                                 lr=args.lr)

    # initialize graph
    dur = []
    
    #Start trainning
    model.train()
    if(mode!=None):
        features_gpu=features.to(device)
        labels_gpu=labels.to(device)
    
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
            if(mode==None):
                feat_in = blocks[0].srcdata['feat']
                labels_out = blocks[-1].dstdata['label']
                #print(feat_in is blocks[0].srcdata['feat'])
                
            else:  
                feat_in = features_gpu[in_nodes.to(device)]
                labels_out = labels_gpu[out_nodes.to(device)]
            t3=time.time()
            print(t3-t2)
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
        print(tS[1:],'\nTotal:',tE-t0)
        
        
    #Finish trainning
    model.eval()

    print("____________________________")
    #acc = evaluate(model, features, labels, test_mask)
    #print("Test accuracy {:.2%}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=4,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    args.neighbor_number = 5
    args.batch_size = 4096
    args.dataset='pubmed'
    print(args)
    #with torch.autograd.profiler.profile(True,True,False) as prof:
    main(args,None)
    #print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    main(args,1)
