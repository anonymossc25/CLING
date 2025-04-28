import numpy as np
import pandas as pd
import sys
import dgl
import time
import torch as th
from scipy.io import mmread
import os
from tqdm import tqdm
import cupy as cp
import psutil
import argparse
os.environ["DGLBACKEND"] = "pytorch"
import torch.nn.functional as F
import torch
import math
import copy
import random
import pymetis
import dgl.data
from dgl import AddSelfLoop
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset,WisconsinDataset,FlickrDataset,RedditDataset,YelpDataset


#-------------------------------------Graph CONSTRUCTION USING data----------------#
totalTime =0
start = time.time()
start1 = time.time()
# file_name, file_extension = os.path.splitext(sys.argv[1])
# print(file_extension)
# suffix_csr = "_output.csr"
# suffix_part = "_reorder.SHEM."
# suffix_part1 = "_reorder.RM."
# file_name = file_name.split("/")
# file_name = file_name[len(file_name)-1]
# out_filename1 = str(file_name) + suffix_csr
# #out_filename2 = str(file_name) + suffix_part + str(sys.argv[2]) + ".csv"
# out_filename3 = str(file_name) + suffix_part1 + str(2) + ".csv"
#print(out_filename2)
if __name__ == "__main__":
    start=time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        #help="Dataset name ('cora', 'citeseer', 'pubmed', 'wisconsin','flickr', 'reddit', 'yelp', 'ogbn-products','ogbn-arxiv').",
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
    elif args.dataset == "yelp":
        data = YelpDataset()
    elif args.dataset == "ogbn-products":
        data = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))    
    elif args.dataset == "ogbn-arxiv":    
        data = AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv"))
    elif args.dataset == "amazon_products":
        load_path = '/data/Dataset/gnn_dataset/amazon_products.dgl'
        data, _ = dgl.load_graphs(load_path)
    elif args.dataset == "cit-net":
        load_path = '/data/Dataset/gnn_dataset/citations_network_graph.dgl'
        data, _ = dgl.load_graphs(load_path)    
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    G = data[0]
    node_feature = G.ndata["feat"]
    node_label = G.ndata["label"]
    node_feature = node_feature.to('cuda')
    #print("node features:",node_feature)
    #print("node labels:",node_label)
    f=G.ndata["feat"].shape[1]
    print("f={}".format(f))
    #finding nodes with feature vectors of all zeros.
    zeros_tensor=torch.zeros(f, device='cuda')

    equality_mask = torch.all(node_feature == zeros_tensor, dim=1)

    # Find the index where all elements are zero
    index = torch.nonzero(equality_mask, as_tuple=True)[0]
    print("Number of feature vector of zeros is {}".format(len(index)))
    size = G.num_edges()
    vertices = G.num_nodes()
    print(f"Number of nodes and edges are {vertices},{size}")

    mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
    print(f"Current memory usage: { (mem_usage)} bytes")
    end = time.time()
    totalTime = totalTime + (end-start)

    print("Data Loading Successfull!!!! \tTime Taken of Loading is :",round((end-start),4), "Seconds")
    mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
    print(f"Current memory usage: { (mem_usage)} GB")

    #----------------------DGL PREPROCESS-----------------------------------#

    Nodes = G.num_nodes()
    Edges = G.num_edges()
    row_ptr=np.array(G.adj_tensors('csr')[0])
    col_idx=np.array(G.adj_tensors('csr')[1])
    row_ptr_s=len(row_ptr)
    col_idx_s=len(col_idx)
    print(row_ptr_s)
    print(col_idx_s)
    end = time.time()
    totalTime = totalTime + (end-start)
    print("Graph Construction Successfull!!!! \tTime Taken :",round((end-start),4), "Seconds")
    #-------------------------------------------Graph Construction is done ----------#
    #---------------cosion similarity--------------------------
    # Extract edges from CSR representation
    #edges = []
    # Convert NumPy array to PyTorch tensor
    row_ptr = torch.tensor(row_ptr)
    col_idx = torch.tensor(col_idx)


    # Move the tensor to GPU
    row_ptr = row_ptr.to('cuda')
    col_idx = col_idx.to('cuda')
    # calculating cosine_similarities
    print("row_ptr size:",len(row_ptr))
    print("col_idx size:",len(col_idx))
    
    #cosine_similarities = torch.Tensor([])
    cosine_similarities = torch.empty(size, device='cuda')
    jaccard_similarity = torch.empty(size, device='cuda')
 
    # for i in range(len(row_ptri) - 1):
    #     start = row_ptr[i].item()
    #     end = row_ptr[i + 1].item()
    #     for j in range(start, end):
    #         src = i
    #         dst = col_idx[j].item()
    #         edges.append((src, dst))
    start_time = time.time()
    #weights=torch.empty(size, device='cuda')
    for i in tqdm(range(len(row_ptr) - 1), desc="processing nodes"):
        start = row_ptr[i].item()
        end = row_ptr[i + 1].item()

        #x = col_idx[start:end]
        #deg = row_ptr[i+1] - row_ptr[i]
        #print("node {} deg {}".format(i,deg))
        #y = node_feature[x]
        #y = y.to('cuda')
        #print("y",y)
        z = node_feature[i]
        #z = z.to('cuda')
        #print("z",z)
        # if i not in index:
        #     cosine_similarities[start:end] = torch.tensor([F.cosine_similarity(z, tensor, dim=0) for tensor in y])
        #     # Use a mask to set negative elements to zero
        #     #cosine_similarities[start:end][cosine_similarities[start:end] < 0] = 0.0
        # else:
        #     cosine_similarities[start:end] = 0.0
        # Neighbors of node i
        src_neighbors = col_idx[start:end]
        src_neighbors = src_neighbors.to(device="cuda")
        #print("Neighbors of source node {} are {}".format(i,src_neighbors))
 
        # Iterate over each neighbor to compute the Jaccard similarity
        for j in range(start, end):
            dst_node = col_idx[j].item()
            #print("destination",col_idx[j])
            cosine_similarities[j] = torch.tensor([F.cosine_similarity(z,node_feature[dst_node], dim=0)])
            #print("cosine similarity:",cosine_similarities[j])

        
            # Get the range of indices in the 'indices' array for the destination node
            dst_start = row_ptr[dst_node].item()
            dst_end = row_ptr[dst_node + 1].item()
        
            # Neighbors of the destination node
            dst_neighbors = col_idx[dst_start:dst_end]
            dst_neighbors = dst_neighbors.to(device="cuda")
            #print("Neighbors of destination node {} are {}".format(dst_node,dst_neighbors))

            # Efficiently find the intersection using broadcasting and logical operations
            intersection = torch.isin(src_neighbors, dst_neighbors).sum().item()
            union = len(src_neighbors) + len(dst_neighbors) - intersection
            
            # Calculate Jaccard similarity
            if union > 0:
                jaccard_similarity[j] = intersection / union
                #print("jaccard", jaccard_similarity[j])
                #jaccard_sim = intersection / union
            else:
                #jaccard_sim = 0.0
                jaccard_similarity[j] = 0.0
                #print("jaccard",jaccard_similarity)
            #print("jaccard_sim: ",jaccard_sim)    
            #jaccard_sim = torch.tensor(jaccard_sim, device="cuda")
            # Convert the scalar to a 1D tensor
            #jaccard_sim = jaccard_sim.unsqueeze(0)
            #print("jaccard_sim of {} and {} is {}".format(i,j,jaccard_sim)) 
            #jaccard_similarity = torch.cat((jaccard_similarity, jaccard_sim), dim=0)
            del dst_neighbors
            #del jaccard_sim
            del dst_node
            #print("j loop end {}".format(j)) print("cosine_similarities: ", cosine_similarities)
    #print("Length of cosine_similarities: ",len(cosine_similarities))

        del src_neighbors 

        #similarities = torch.where(similarities < 0, torch.tensor(0.0), similarities)
        #similarities = th.matmul(y,z)
        #weights[row_ptr[i]:row_ptr[i+1]]=similarities
        #similarities = similarities.to('cuda')
        #print("Length of similarities",len(similarities))
        #print("Similarities array corresponding to node{} is{}".format(i,similarities))
        #cosine_similarities = torch.cat((cosine_similarities,similarities))
        #del y
        del z
        #del x
        #del deg
    cosine_similarities = torch.clamp(cosine_similarities, min=0)
    #print("cosine_similarities: ", cosine_similarities)
    #print("Length of cosine_similarities: ",len(cosine_similarities))


    #print("jaccard_similarity: ",jaccard_similarity)
    #print("Length of jaccard_similarity",len(jaccard_similarity))
    #print("Jaccard_time: ",Jaccard_time_end - Jaccard_time_start)
    #weight_vector1 = cosine_similarities + jaccard_similarity
    #torch.set_printoptions(threshold=torch.inf)
    weight_vector1 = torch.add(cosine_similarities,jaccard_similarity)
    #weight_vector1 = torch.add(weights,jaccard_similarity)
    weight_vector = weight_vector1 * 50
    weight_vector = torch.round(weight_vector)
    weight_vector = weight_vector.to(torch.int64)
    print("weight_vector",weight_vector)
    #print("Length of weight_vector", len(weight_vector))
  
    #cosine_similarities = torch.where(cosine_similarities < 0, torch.tensor(0.0), cosine_similarities)
    
    end_time = time.time()
    #print(weights)
    #print("Length of weight: ",len(weights))
    #print("Weight tensor: ",weights)
    print("Weight calculation times: ",end_time - start_time)
    '''
    cosine_similarities = cosine_similarities.tolist()
    file_path4 = 'cluster_id.txt'
    with open(file_path4, "w") as file:
        for value in cosine_similarities:
        file.write(f"{value}\n")
    '''    

    # Convert edge list to a tensor for batch processing
    #edges = torch.tensor(edges)
    #print("Edges tensor has been created")

    # Extract features for source and destination nodes
    #src_features = G.ndata['feat'][edges[:, 0]]
    #dst_features = G.ndata['feat'][edges[: ,1]]
    # src_features = node_features[edges[:, 0]]
    # dst_features = node_features[edges[:, 1]]
    #print(type(src_features))
    #print(src_features.shape)
    # Number of elements in the tensor
    #num_elements = src_features.numel()

    # Size of each element in bytes
    #element_size = src_features.element_size()

    # Total memory in bytes
    #total_memory_bytes = num_elements * element_size

    # Convert to gigabytes
    #total_memory_gb = total_memory_bytes / (1024 ** 3)  # Dividing by 2^30 for GB
    #print(total_memory_gb)

    
    # Compute cosine similarity for all edges
    #cosine_similarities = F.cosine_similarity(src_features, dst_features)
    #print("cosine_similarities: ",cosine_similarities)
    #print("size cosine_similarities: ", len(cosine_similarities))
    # Assign the edge weights to the 'weight' attribute of the graph
    weight_vector = weight_vector.cpu()
    G.edata['weight'] = weight_vector
    xadj = row_ptr.tolist()  # The xadj array in PyMetis (cumulative degree list)
    adjncy = col_idx.tolist()  # The adjacency list in PyMetis
    #adjwgt = weight_arr.tolist()# The edge weights in PyMetis
    adjwgt = weight_vector.tolist() 
    # nopart = int(sys.argv[2])
    nopart = 20
    torch.set_printoptions(threshold=torch.inf)
    print("Start Partitioning Weight_graph.....")
    start = time.time()
    node_parts_weight = dgl.metis_partition_assignment(G,nopart)
    #n_cuts, membership = pymetis.part_graph(nopart, xadj=iptr, adjncy=indx, eweights=weights)
    # cut, membership = pymetis.part_graph(nopart, adjacency=None, xadj=xadj, adjncy=adjncy, vweights=None, eweights=adjwgt)
    # cut, membership = pymetis.part_graph(5, xadj=xadj, adjncy=adjncy, eweights=adjwgt)
    #RG = dgl.reorder_graph(G, node_permute_algo='metis', edge_permute_algo='dst', permute_config={'k':nopart})
    end = time.time()
    totalTime = totalTime + (end-start)
    print("Partition is Done !!!!!\t Time of Partition is :",round((end-start),4), "Seconds")
    mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
    print(f"Current memory usage: { (mem_usage)} bytes")
    #print("node_parts_weight: ",node_parts_weight)
    # print("cuts: ", cut)
    # print("membership: ", membership)


    # node_parts_weight = np.sort(node_parts_weight)
    # node_parts = np.sort(node_parts)

    # row_ptr=np.array(RG.adj_sparse('csr')[0])
    # col_idx=np.array(RG.adj_sparse('csr')[1])
    # row_ptr_s=len(row_ptr)
    # col_idx_s=len(col_idx)


    #del g_row_ptr
    #del g_col_idx
    del row_ptr
    del col_idx
    #del g_weight_arr
    #del g_sum
    #del weight_arr
    del weight_vector
    del weight_vector1
    #del sum
    #cp.cuda.runtime.free(intptr_t temp_arr)
    cp._default_memory_pool.free_all_blocks()
    end1 = time.time()
    print("Preprocess Successfull!!!! \tTime Taken of Prepr weight_vector1 = torch.add(cosine_similarities,jaccard_similarity)ocess is :",round((end1-start1),4), "Seconds")
    # Get unique values and the inverse indices (where each value was found)
    unique_values, inverse_indices = node_parts_weight.unique(return_inverse=True)

    # Determine the number of unique values
    num_unique_values = unique_values.size(0)

    # Create a list to hold the indices for each unique value
    clusters = []

    # Iterate through unique values and collect indices
    for unique_value in unique_values:
        # Get the indices of the current unique value
        indices = (inverse_indices == unique_value).nonzero(as_tuple=True)[0]
        clusters.append(indices)
    #clusters = torch.tensor(clusters)
    #clusters = clusters.to('cuda')

    # Find the maximum number of indices to pad the result
    #max_length = max(len(indices) for indices in indices_list)

    # Create a 2D tensor with padding
    #result_tensor = torch.full((num_unique_values, max_length), fill_value=-1, dtype=torch.long)

    # Populate the 2D tensor with the indices
    #for i, indices in enumerate(indices_list):
        #result_tensor[i, :len(indices)] = indices

    #print(type(clusters))
    #print("clusters")
    #print(clusters)
    #representative = []
    representative = torch.empty(0, f)
    #representative = representative.to('cuda')
    node_feature = node_feature.cpu()
    for j, row in enumerate(clusters):
        #print(j)
        y=node_feature[row]
        summ=torch.zeros(f)
        for p in y:
            summ+=p
        r=summ/len(y)
        representative = torch.cat((representative, r.view(1, -1)), dim=0)
    representative = representative.tolist()
        #del y
        #del summ
        #del r
    #representative = representative.tolist();
    #PRINTING REPRESENTIVE OF EACH CLUSTER
    
    # for j, row in enumerate(representative):
    #     print("representative of {} is".format(j))
    #     print(row)
    #     print("Length of representative",len(row))
    file_path2 = '/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster/representative.npy'
    # Convert the list to a NumPy array
    np_representative = np.array(representative)

    # Save the NumPy array to a .npy file
    np.save(file_path2, np_representative)
    node_parts_weight = node_parts_weight.tolist()
    file_path1 = '/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster/cluster_id.txt'
    with open(file_path1, "w") as file:
        for value in node_parts_weight:
            file.write(f"{value}\n")

