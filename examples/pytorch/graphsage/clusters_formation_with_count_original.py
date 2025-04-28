import argparse
import os
import sys
import time
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import numpy as np
import torch.nn.functional as F
import torch
import math
import copy
import random
import dgl.data
from dgl import AddSelfLoop
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset,WisconsinDataset,FlickrDataset,RedditDataset

if __name__ == "__main__":
    start=time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'citeseer', 'pubmed', 'wisconsin','flickr', 'reddit','ogbn-products','ogbn-arxiv').",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=10,
        help="Number of cluster",
    )

    parser.add_argument(
        "--dt",
        type=str,
        default="float",
        help="data type(float, bfloat16)",
    )
    args = parser.parse_args()
    print(f"Training with DGL built-in GraphConv module.")

    # load and preprocess dataset
    if args.dataset == "cora":
        data = CoraGraphDataset()
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset()
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset()
    elif args.dataset == "wisconsin":
        data = WisconsinDataset()    
    elif args.dataset == "flickr":
        data = FlickrDataset()
    elif args.dataset == "reddit":
        data = RedditDataset()   
    elif args.dataset == "ogbn-products":
        data = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))    
    elif args.dataset == "ogbn-arxiv":    
        data = AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv"))
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    g = data[0]
    print("number of edges{}",g.num_edges())
    g = dgl.add_reverse_edges(g)
    print(f"Number of categories: {data.num_classes}")
    print(
        "load {} takes {:.3f} seconds".format(data, time.time() - start)
    )
    print("|V|={}, |E|={}".format(g.num_nodes(), g.num_edges()))
    print(g)
    node_feature=g.ndata["feat"]
    node_label=g.ndata["label"]
    print("node features:",node_feature)
    print("node labels:",node_label)
    '''
    if args.undirected:
        sym_g = dgl.to_bidirected(g, readonly=True)
        for key in g.ndata:
            sym_g.ndata[key] = g.ndata[key]
        g = sym_g
    '''
    start=time.time()
    f=g.ndata["feat"].shape[1]
    print("f={}".format(f))
    #finding nodes with feature vectors of all zeros.
    zeros_tensor=torch.zeros(f)
    equality_mask = torch.all(node_feature == zeros_tensor, dim=1)

    # Find the index where all elements are zero
    index = torch.nonzero(equality_mask, as_tuple=True)[0]
    print("Number of feature vector of zeros is {}".format(len(index)))

    #print("Index of 1D tensor in 2D tensor where all elements are zero:", index)
    #print(len(index))
    #print(type(index))
    file_path3 = 'index.pt'
    torch.save(index, file_path3)

    index=index.tolist()
    #file_path3 = 'index.txt'

    #finding  distinct node for each cluster
    x = torch.empty(0, f)
    p=args.num_clusters*40
    for i in range(p):
         x = torch.cat((x,node_feature[i].view(1, -1)), dim=0)
    data_normalized = F.normalize(x, p=2, dim=1)
    #Calculate the cosine similarity matrix
    similarity_matrix = torch.mm(data_normalized, data_normalized.t())
    my_list = []
    rows,columns = similarity_matrix.shape
    for i in range(rows):
        for j in range(columns):
            element = similarity_matrix[i,j]
            if (i < j):
                my_list.append(element)
    sorted_list = sorted(my_list)

    # Get the least k elements
    least_k_elements = sorted_list[:args.num_clusters]
    least_k_elements = torch.tensor(least_k_elements)
    #print("Least similarity")
    #print(least_k_elements)
    #x = torch.empty(0, 2)
    #x = torch.zeros(2,dtype=torch.int64)
    x=torch.Tensor([])
    #x = torch.empty((), dtype=torch.int64)
    #print("x:",x)
    for i in least_k_elements:
        indices = torch.where(similarity_matrix == i)
        #print(indices)
        #print(type(indices))
        merge=torch.cat((indices[0],indices[1]),dim=0)
        #print("merge",merge)
        #x = torch.cat((x,indices[0].view(1, -1)), dim=0)
        x = torch.cat((x,merge), dim=0)
        #print("indices of {} is".format(i))
        #print(indices)
        #print(type(indices))
    #print("index of Least similar nodes")
    #indices_to_remove = [0,1]
    #x = torch.tensor([x[i] for i in range(len(x)) if i not in indices_to_remove])
    x = torch.unique(x)
    x = x.type(torch.int64)
    #print("x:",x)
    y=x[ :args.num_clusters]
    #print("y",y)
    x=node_feature[y]
    clusters = y.view(-1, 1)
    #print("initial cluster")
    #print(y)
    clusters=clusters.tolist()
    #x=node_feature[[4,9,33,44,67]]
    #print("node feature of 5 nodes")
    #print(x)
    #clusters=[[4],[9],[33],[44],[67]]
    cluster_id=torch.Tensor([])
    for i in range(g.num_nodes()):    
        if i not in index:
            #for i in range(50):
            similarities = torch.tensor([F.cosine_similarity(node_feature[i],tensor, dim=0) for tensor in x])
            #print("similarity vector")
            #print(similarities)
            largest_element = torch.max(similarities)
            indices = torch.where(similarities == largest_element)
            #print("indices",indices)
            #ii,=indices
            ii=indices[0]
            if(len(ii)>1): #more than 1 elements has same higher similarity value
                cluster_id=torch.cat((cluster_id,torch.tensor([ii[0]])))
                if i not in clusters[ii[0]]:
                   clusters[ii[0]].append(i)
            else:       
                cluster_id=torch.cat((cluster_id,ii))
            #cluster_id[i]=ii
                if i not in clusters[ii]:
                     clusters[ii].append(i)
            #print("cluster forming")     
            #print("Similarity of node {} with respect to 5 cluster are".format(i))
            #print(similarities)
            #print("highest similarity is {}".format(largest_element))
            #print("index is {}".format(ii))
            #print(type(ii))
            #for j in clusters:
                #print(j)
        else:
            cluster_id=torch.cat((cluster_id,torch.tensor([-1]))) #cluster is -1 for nodes having feature vector of all zeros.
    #for j in clusters:
        #print(j)
    file_path = '/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster/clusters.txt'

    # Open the file for writing and save the 2D list
    with open(file_path, 'w') as file:
        for row in clusters:
            file.write(' '.join(map(str, row)) + '\n')
    '''        
    file_path4 = 'cluster_id.txt'

    # Open the file for writing and save the 2D list
    with open(file_path4, 'w') as file:
        for row in cluster_id:
            file.write(' '.join(map(str, row)) + '\n')        
    '''        
    '''        
    loaded_2d_list = []
    # Open the file for reading and load the 2D list
    with open(file_path, 'r') as file:
        for line in file:
            row = line.strip().split()  # Remove leading/trailing whitespace and split into elements
            row = [int(item) for item in row]  # Convert the strings to integers
            loaded_2d_list.append(row)
    cluster_id=cluster_id.to(torch.int64)
    cluster_id = cluster_id.tolist()
    '''
    cluster_id = cluster_id.to(dtype=torch.int64)
    file_path1 = '/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster/cluster_id.pt'
    # Save the tensor to a file
    torch.save(cluster_id, file_path1)
    # Load the tensor from the file
    #loaded_tensor = torch.load(file_path1)
    #print("cluster id")
    #print(cluster_id)
    cluster_id = cluster_id.tolist()
    file_path4 = '/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster/cluster_id.txt'
    with open(file_path4, "w") as file:
        for value in cluster_id:
            file.write(f"{value}\n")

    #print("#" * 120)
    #FINDING THE REPRESENTATIVE FOR EACH CLUSTER
    representative = torch.empty(0, f)
    for j, row in enumerate(clusters):
        #print(j)
        y=node_feature[row]
        summ=torch.zeros(f)
        for p in y:
            summ+=p
        r=summ/len(y)
        representative = torch.cat((representative, r.view(1, -1)), dim=0)
    #PRINTING REPRESENTIVE OF EACH CLUSTER
    #for j, row in enumerate(representative):
        #print("representative of {} is".format(j))
        #print(row)
    file_path2 = '/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster/representative.pt'
    torch.save(representative, file_path2)
    print("Time for clustering is {}".format(time.time()-start))
