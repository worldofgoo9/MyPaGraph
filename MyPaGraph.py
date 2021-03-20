# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 09:56:09 2021

@author: worldofgoo9
"""

import dgl
import torch
import sys
import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import numpy as np
import random
#import networkx as nx
#import matplotlib.pyplot as plt
import time

class Init:
    def __init__(self):
        pass
    
    def __GetHopNeighbors__(g,nodes,hop):
        result = set(nodes.numpy().tolist())
        
        last_hop = set(nodes.numpy().tolist())
        
        for i in range(hop):
            current_neighbors = set()
            for n in last_hop:
                current_neighbors.update(g.predecessors(n).numpy().tolist())
                
            last_hop = current_neighbors.difference(result)
            result.update(current_neighbors)
            
        return torch.tensor(list(result),dtype = torch.long,requires_grad = False)

            
    def DivideGraph(g,num,hop,target_nodes = None,PV_need = True,print_info=True,information_print_gap = -1):
        print(hop,num)
        assert (hop > 0) and (num > 0),"hop and num must greater than zero" 
        
        if(target_nodes == None):
            target_nodes = g.nodes().numpy().tolist()
            #That means take all the nodes in graph as target nodes.
        if(information_print_gap <= 0):
            information_print_gap = int(0.1 * len(target_nodes))
        
        TVavg = len(target_nodes) / num
        TV = []
        TV_len = []
        PV = []
        PV_len = []
        for _ in range(num):
            TV.append(set())
            PV.append(set())
            TV_len.append(0)
            PV_len.append(1e-12)
            
        '''
        Using TV_len and PV_len to record number of nodes in TV and PV in order
        to avoid redundant caculations.
        Elements in PV_len are set in 1e-12 to avoid "divide by zero" problem
                
        (About TV and PV )In the paper:
        𝑇𝑉𝑖 represents the train vertex set already assigned to the 𝑖-th
        partition. 𝐼𝑁 (𝑉𝑡) denotes the 𝐿-hop in-neighbor set of train
        vertex 𝑣𝑡. 𝑃𝑉𝑖 controls the workload balance, and denotes
        the total number of vertices in the 𝑖-th partition, including
        the replicated vertices. 𝑣𝑡
        is most likely to be assigned to
        a partition which has smallest 𝑃𝑉 .
        To achieve computation balance, we set 𝑇𝑉𝑎𝑣𝑔 as |𝑇𝑉 | /𝐾
        , which indicates
        that all partitions will get almost the same number of train
        vertices.
        '''
         
        # Set Score numpy
        scores = np.zeros(num,dtype = np.float32)
        
        # Set random seed
        random.seed(time.time())
        
        TR=[0.0,0.0,0.0,0.0,0.0]
        TS=0.0
        # Begin
        t_start = time.time()
        for epoch,(node) in enumerate(target_nodes):
            if(epoch % information_print_gap == 0):   
                print(f"Epoch:{epoch}  ,Time:{time.time()-t_start} seconds.")
                

            #Get L-hop neighbors.
            
            with torch.no_grad():                
                result = {node}
                last_hop = {node}
                for i in range(hop):
                    current_neighbors = set()
                    t00=time.time()
                    for n in last_hop:
                        current_neighbors.update(g.in_edges(n)[0].numpy().tolist())
                    t01=time.time()
                    TS+=t01-t00
                    last_hop = current_neighbors.difference(result)
                    result.update(current_neighbors)
                #input_nodes = torch.tensor(result)
            
                neighbor_set = result
            # neighbor_set denotes IN(Vt) in paper.

            
            
            for i in range(num):
                scores[i] = len(TV[i].intersection(neighbor_set)) \
                 * (TVavg-TV_len[i]) / PV_len[i]
                #scores[i] = 0.0 only for test(which means randomly divide the graph).

                
            max_score = scores.max()
            max_score_mask = (scores == max_score)
            max_score_num = int(max_score_mask.sum())
            max_score_index = (np.arange(0,num,1))[max_score_mask]
            

            if(max_score_num > 1):
                # If there are several elements with max score(Most likely to be 0.0).
                # Randomly choose one of them.
                random_choose = random.randint(0,max_score_num-1)
                divide_index = max_score_index[random_choose]    
            else:
                divide_index = scores.argmax()

            TV[divide_index].add(node)
            TV_len[divide_index] += 1
            PV[divide_index].update(neighbor_set)
            PV_len[divide_index] = max(1e-12,len(PV[divide_index]))



            


        # Print summary
        if(print_info):
            for i in range(num):
                print(f'Partition {i}:          {len(TV[i])} target nodes ,\
                  {len(PV[i])-len(TV[i])} redundant nodes')
        
        # Transform into tensor  
        with torch.no_grad():
            for i in range(num):
                TV[i] = torch.tensor(list(TV[i]),dtype = torch.long)
                PV[i] = torch.tensor(list(PV[i]),dtype = torch.long)
        
        
        if(PV_need):    
            return TV , PV
        else:
            return TV
    
    def CreateStorageTools(g,device_list,TV_list,PV_list):
        assert len(device_list) == len(TV_list) == len(PV_list)
        
        
class Storage:
    def __init__(self,g,data,cache_rate,nodes,gpu,cpu='cpu'):
        assert cache_rate >= 0.0 and cache_rate <= 1.0,"Cache rate should be a number between 0.0 and 1.0"
        
        ''' Init '''
        self.nnum_all = g.number_of_nodes()
        self.gpu = gpu
        self.cpu = cpu
        self.cache_rate = cache_rate
        self.nodes = nodes
        self.in_gpu = int(len(nodes)*cache_rate)
        self.in_cpu = len(nodes)-self.in_gpu
        self.featnum = len(data)
        
        self.store_map = torch.zeros([self.nnum_all,2],dtype=torch.long,requires_grad=False)-1
        # Init to -1
        self.store_data = [dict(),dict()]

        #store_map[:,0] indicate where are stored (0:cpu,1:gpu)
        #store_map[:,1] indicate the actual index of corespond feature stored in tensor
        
        for name in data:
            shape_gpu = torch.tensor(data[name].shape)
            shape_cpu = torch.tensor(data[name].shape)
            shape_gpu[0] = self.in_gpu
            shape_cpu[0] = self.in_cpu
            dt = data[name].dtype
            #self.store_gpu[name] = torch.zeros(shape_gpu,dtype=dt,
            #                                   device=self.gpu,requires_grad=False)
            #self.store_cpu[name] = torch.zeros(shape_cpu,dtype=dt,
            #                                   device=self.cpu,requires_grad=False)
            
        
        
        ''' Start to store features '''  
        # Firstly , rank nodes by their out-degree.Choose nodes with top degree.
        OutDegree = g.out_degrees()[nodes]
        OrderedIndex = nodes[(torch.sort(OutDegree,descending=True)[1])]
        NodesGpu = torch.sort(OrderedIndex[0:self.in_gpu])[0]
        NodesCpu = torch.sort(OrderedIndex[self.in_gpu:])[0]
        
        # Update store_map.
        self.store_map[NodesGpu,0] = 1
        self.store_map[NodesCpu,0] = 0
        self.store_map[NodesGpu,1] = torch.arange(0,self.in_gpu)
        self.store_map[NodesCpu,1] = torch.arange(0,self.in_cpu)
        
        # Store features.
        for name in data:
            self.store_data[1][name] = data[name][NodesGpu].to(self.gpu)
            self.store_data[0][name] = data[name][NodesCpu].to(self.cpu)
            assert len(self.store_data[1][name])+len(self.store_data[0][name])==len(self.nodes)
            
            
    def Query(self,fname,nodes,print_info=False):

        
        # Init "result" tensor for return.
        result_shape = torch.tensor(self.store_data[0][fname].shape)
        result_shape[0] = len(nodes)
        result_type = self.store_data[0][fname].dtype
        result = torch.zeros(result_shape.numpy().tolist(),
                             dtype=result_type,device=self.gpu,requires_grad=False)
        
        # Get "stored" map and mask.
        stored_map = self.store_map[nodes]
        gpu_mask = (stored_map[:,0] == 1)
        cpu_mask = (stored_map[:,0] == 0)
        assert stored_map.min() >= 0,"Querying a node that was not stored!"
        if(print_info):
            print(f"{gpu_mask.sum()} are cached in gpu, {cpu_mask.sum()} are stored in cpu.")
        
        
        # Insert features into "result" tensor.
        result[gpu_mask] = self.store_data[1][fname][(stored_map[gpu_mask][:,1])].to(self.gpu)
        result[cpu_mask] = self.store_data[0][fname][(stored_map[cpu_mask][:,1])].to(self.gpu)

        # Finnal ,return.        
        return result        
    
'''
class Storage_Old:
    def __init__(self,g,device_list,PV_list,TV_list = None):
        self.graph = g.to('cpu')
        self.device_list = device_list
        self.TV_list = TV_list
        self.PV_list = PV_list
        self.hash_map = []
        self.data = []
        for i in range(len(device_list)):
            with torch.no_grad():
                self.hash_map.append(torch.zeros([self.graph.num_nodes()],dtype=torch.long))
                self.hash_map[i][:] = -1
                #Init as -1
                self.data.append(dict())
                
    def StoreIntoDevices(self,cache_rate,feat_name_list):
        assert cache_rate >= 0.0 and cache_rate <= 1.0
        for i,(device) in enumerate(self.device_list):
            cache_num = int(len(self.PV_list[i])*cache_rate)
            PV_degrees = self.graph.out_degrees()[self.PV_list[i]]                        
            ordered_index = self.PV_list[i][torch.sort(PV_degrees,descending = True)[1]]
            with torch.no_grad():
                store_index = torch.sort(ordered_index[0:cache_num].clone())[0]
            self.hash_map[i][store_index] = torch.arange(0,cache_num)
            
            for fname in feat_name_list:
                self.data[i][fname] = self.graph.ndata[fname][store_index].to(device)
    
    def Query(self,device_index,nodes,fname):
        
        stored_map = self.hash_map[device_index][nodes]
        stored_mask = (stored_map >= 0)
        not_stored_mask = (stored_mask == False)
        print(f"{stored_mask.sum()} are cached , {not_stored_mask.sum()} are not cached .")
        
        shape = list(self.graph.ndata[fname].shape)
        shape[0] = len(nodes)
                
        assert shape[0] == stored_mask.sum() + not_stored_mask.sum(),"Check the argument 'nodes'"
        result = torch.zeros(shape,
                             dtype = self.graph.ndata[fname].dtype,
                             device = self.device_list[device_index],
                             requires_grad = False)
        
        result[stored_mask] = self.data[device_index][fname][stored_map[stored_mask]]
        result[not_stored_mask] = self.graph.ndata[fname][nodes[not_stored_mask]].to(self.device_list[device_index])
        
        return result
'''        
        