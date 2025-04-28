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
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset,WisconsinDataset,FlickrDataset,RedditDataset

#256 = 256


Weight_graph = cp.RawKernel(r'''
__device__ __forceinline__ int Search (int skey , int *neb, int sizelist)
{
    int total = 0;
    if(skey < neb[0] || skey > neb[sizelist])
    {
        return 0;
    }
    else if(skey == neb[0] || skey == neb[sizelist])
    {
        return 1;
    }
    else
    {
        int lo = 1;
        int hi = sizelist-1;
        int mid=0;
        while( lo <= hi)
        {
            mid = (hi+lo)/2;
            //printf("\nskey :%d , mid : %d ",skey,neb[mid]);
            if( neb[mid] < skey){lo=mid+1;}
            else if(neb[mid] > skey){hi=mid-1;}
            else if(neb[mid] == skey)
            {
                total++;
                break;
            }
        }
    }
    return total;
}

extern "C" __global__
void Weight_graph(unsigned long long int *g_col_index, unsigned long long int *g_row_ptr ,unsigned long long int *g_sum, unsigned long long int *g_weight_arr )
{
    //int id = threadIdx.x + blockIdx.x * blockDim.x ; //Define id with thread id
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ int start;
    __shared__ int end;
    unsigned long long int triangle;
    unsigned long long int count;
    __shared__ int neb[256];
    __shared__ unsigned long long int s_sum[256];
    //int start = g_row_ptr[bid];
    //int end = g_row_ptr[bid+1]-1;
    //int index = reordered_array[bid];
    if(tid ==0)
    {
        //triangle = 0;
        start = g_row_ptr[bid];
        end = g_row_ptr[bid+1]-1;
    }
    __syncthreads();
    int size_list1 = end - start;
    if(size_list1 < 1)
    {
        g_sum[bid] = 0;
    }
    else
    {
    //if(size_list1 ==0 ) return;
            if(size_list1 < 256)
            {
                if(tid <= size_list1)
                {
                    neb[tid] = g_col_index[tid+start];
                }
                __syncthreads();
                for( int i = 0; i <= size_list1; i++)
                {
                    count = 0;
                    int start2 = g_row_ptr[neb[i]];
                    int end2 = g_row_ptr[neb[i]+1]-1;
                    int size_list2 = end2 - start2;
                    int M = ceil((float)(size_list2 +1)/256);
                    //#pragma unroll
                    for( int k = 0; k < M; k++)
                    {
                        int id = 256 * k + tid;
                        if(id <= size_list2)
                        {
                            int result = 0;
                            result = Search(g_col_index[id+start2],neb,size_list1);
                            //printf("\nedge(%d , %d) : %d , tid : %d, size_list1 :%d , size_list2: %d, start2 :%d , end2 :%d skey:%d, neb[0]:%d ,neb[%d]:%d",bid, neb[i], result,tid,size_list1+1,size_list2+1,start2,end2,g_col_index[id+start2],neb[0],size_list1,neb[size_list1]);
                            //atomicAdd(&g_sum[0],result);
                            //printf("\nedge(%d , %d) src : %d dst :%d ", bid,neb[i],size_list1+1,size_list2+1);
                            triangle += result;
                            count = count + result;
                        }
                    }
                    __syncthreads();
                    s_sum[tid] = count;
                    __syncthreads();
                    if (tid == 0)
                    {
                        unsigned long long int edge_sum = 0;

                        for (int j = 0; j < 256; j++)
                        {
                            edge_sum += s_sum[j];
                        }
                        g_weight_arr[i + start] = edge_sum+1;
                        //printf("\n edge (%d, %d) Edge_sum : %d", bid, neb[i], edge_sum);
                    }
                }
            }
            else
            {
                int N = ceil((float)(size_list1 +1)/ 256);
                int remining_size = size_list1;
                int size = 256-1;
                for( int i = 0; i < N; i++)
                {
                    count = 0;
                    int id = 256 * i + tid;
                    if( remining_size > size)
                    {
                        if(id <= size_list1)
                        {
                            neb[tid] = g_col_index[id+start];
                            //printf(" neb : %d", neb[tid]);
                        }
                        __syncthreads();
                        for( int j = start; j <= end; j++)
                        {
                            int start2 = g_row_ptr[g_col_index[j]];
                            int end2 = g_row_ptr[g_col_index[j]+1]-1;
                            int size_list2 = end2 - start2;
                            int M = ceil((float)(size_list2 +1)/256);

                            for( int k = 0; k < M; k++)
                            {
                                int tempid = 256 * k + tid;
                                if(tempid <= size_list2)
                                {
                                    int result = 0;
                                    result = Search(g_col_index[tempid+start2],neb,size);
                                    //printf("\nedge(%d , %d) : %d , tid : %d, size_list1 :%d , size_list2: %d, start2 :%d , end2 :%d, id :%d, skey :%d, N:%d, I:%d, remining_size:%d, size:%d, neb[0]:%d, neb[%d]:%d if ",bid, g_col_index[j], result,tid,size_list1+1,size_list2+1,start2,end2,id,g_col_index[tempid+start2],N,i,remining_size,size,neb[0],size,neb[size]);
                                    //atomicAdd(&g_sum[0],result);
                                    //printf("\nedge(%d , %d) src : %d dst :%d ", bid,g_col_index[j],size_list1+1,size_list2+1);
                                    triangle += result;
                                    count = count + result;
                                }
                            }
                        }
                        __syncthreads();
                        remining_size = remining_size-(size+1);
                    }
                    else
                    {

                        if(id <= size_list1)
                        {
                            neb[tid] = g_col_index[id+start];
                            //printf(" neb : %d", neb[tid]);
                        }
                        __syncthreads();
                        for( int j = start; j <= end; j++)
                        {
                            int start2 = g_row_ptr[g_col_index[j]];
                            int end2 = g_row_ptr[g_col_index[j]+1]-1;
                            int size_list2 = end2 - start2;
                            int M = ceil((float)(size_list2 +1)/ 256);

                            for (int k = 0; k < M; k++)
                            {
                                int tempid = 256 * k + tid;
                                if(tempid <= size_list2)
                                {
                                    int result = 0;
                                    result = Search(g_col_index[tempid+start2],neb,remining_size);
                                    //printf("\nedge(%d , %d) : %d , tid : %d, size_list1 :%d , size_list2: %d, start2 :%d , end2 :%d, id :%d, skey :%d, N:%d, I:%d neb[0]:%d, neb[%d]:%d, else",bid, g_col_index[j], result,tid,size_list1+1,size_list2+1,start2,end2,id,g_col_index[tempid+start2],N,i,neb[0],remining_size,neb[remining_size]);
                                    //atomicAdd(&g_sum[0],result);
                                    //printf("\nedge(%d , %d) src : %d dst :%d ", bid,g_col_index[j],size_list1+1,size_list2+1);
                                    triangle += result;
                                    count = count + result;
                                }
                            }
                        }
                    }
                    __syncthreads();
                    s_sum[tid] = count;
                    __syncthreads();
                    if (tid == 0)
                    {
                        unsigned long long int edge_sum = 0;

                        for (int j = 0; j < 256; j++)
                        {
                            edge_sum += s_sum[j];
                        }
                        //g_sum[bid] = block_sum;
                        g_weight_arr[i + start] = edge_sum+1;
                        //printf("\n edge (%d, %d) Edge_sum : %d", bid, neb[i], edge_sum);
                    }
                }
            }
    //	atomicAdd(&g_sum[0],triangle);
    /*
        s_sum[tid] = triangle;
    __syncthreads();
     if (tid == 0)
     {
         unsigned long long int block_sum = 0;

         for (int i = 0; i < 256; i++)
         {
             block_sum += s_sum[i];
         }
         g_sum[bid] = block_sum;
     }
     */
    }

        // if(tid ==0)
        // g_sum[bid] = triangle;
    //	printf("%llu",triangle);
}

'''
                            , 'Weight_graph')


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
    G = data[0]
    node_feature = G.ndata["feat"]
    node_label = G.ndata["label"]
    node_feature = node_feature.to('cuda')
    #print("node features:",node_feature)
    #print("node labels:",node_label)
    f=G.ndata["feat"].shape[1]
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
    #--------------------------------GPU CODE-----------------------------------------##
    # Create CuPy arrays for input and output data
    g_row_ptr = cp.asarray(row_ptr)
    g_col_idx = cp.asarray(col_idx)
    g_weight_arr = cp.empty_like(col_idx)
    #g_sum = cp.zeros(1)
    g_sum = cp.empty_like(row_ptr)

    # Configure the kernel launch parameters
    block_size = 256
    grid_size = (row_ptr_s + block_size - 1) // block_size
    #grid_size = Nodes
    print(grid_size)
    # Launch the kernel
    Weight_graph((Nodes,), (block_size,), (g_col_idx, g_row_ptr, g_sum, g_weight_arr))

    #sum = cp.cumsum(g_sum)
    #N = sum[len(sum)-1]
    #sum = g_sum.get()
    weight_arr = g_weight_arr.get()

    print("weight_arr: ",weight_arr)
    print("size weight_arr: ",len(weight_arr))
    #print(int(N/6))

    weight_arr = th.tensor(weight_arr)

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
    # for i in range(len(row_ptri) - 1):
    #     start = row_ptr[i].item()
    #     end = row_ptr[i + 1].item()
    #     for j in range(start, end):
    #         src = i
    #         dst = col_idx[j].item()
    #         edges.append((src, dst))
    cosine_start = time.time()
    #weights=torch.empty(size, device='cuda')
    for i in range(len(row_ptr) - 1):
        x = col_idx[row_ptr[i]:row_ptr[i+1]]
        deg = row_ptr[i+1] - row_ptr[i]
        #print("node {} deg {}".format(i,deg))
        y = node_feature[x]
        #y = y.to('cuda')
        #print("y",y)
        z = node_feature[i]
        #z = z.to('cuda')
        #print("z",z)
        cosine_similarities[row_ptr[i]:row_ptr[i+1]] = torch.tensor([F.cosine_similarity(z, tensor, dim=0) for tensor in y])
        #similarities = torch.where(similarities < 0, torch.tensor(0.0), similarities)
        #similarities = th.matmul(y,z)
        #weights[row_ptr[i]:row_ptr[i+1]]=similarities
        #similarities = similarities.to('cuda')
        #print("Length of similarities",len(similarities))
        #print("Similarities array corresponding to node{} is{}".format(i,similarities))
        #cosine_similarities = torch.cat((cosine_similarities,similarities))
        del y
        del z
        del x
        del deg
    cosine_similarities = torch.clamp(cosine_similarities, min=0)    
    #cosine_similarities = torch.where(cosine_similarities < 0, torch.tensor(0.0), cosine_similarities)
    
    cosine_end = time.time()
    #print(weights)
    #print("Length of weight: ",len(weights))
    #print("Weight tensor: ",weights)
    print("cosine_similarities times: ", cosine_end - cosine_start)
    print("cosine_similarities: ", cosine_similarities)
    print("Length of cosine_similarities: ",len(cosine_similarities))
    '''
    cosine_similarities = cosine_similarities.tolist()
    file_path4 = 'cluster_id.txt'
    with open(file_path4, "w") as file:
        for value in cosine_similarities:
        file.write(f"{value}\n")
    '''    

    # for i in range(len(row_ptr) - 1):
    #     x = col_idx[row_ptr[i]:row_ptr[i+1]]
    #     for j in x:
    #         
    Jaccard_time_start = time.time()
    # Initialize a list to store Jaccard similarities for each edge
    jaccard_similarity = torch.empty(size, device='cuda')
    #torch.set_printoptions(threshold=torch.inf)

    # Iterate over each node to find edges
    for i in range(len(row_ptr)-1):
        # Get the range of indices in the 'indices' array for node i
        start = row_ptr[i].item()
        end = row_ptr[i + 1].item()
    
        # Neighbors of node i
        src_neighbors = col_idx[start:end]
        src_neighbors = src_neighbors.to(device="cuda")
        #print("Neighbors of source node {} are {}".format(i,src_neighbors))
 
        # Iterate over each neighbor to compute the Jaccard similarity
        for j in range(start, end):
            dst_node = col_idx[j].item()
        
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
                #jaccard_sim = intersection / union
            else:
                #jaccard_sim = 0.0
                jaccard_similarity[j] = 0.0
            #print("jaccard_sim: ",jaccard_sim)    
            #jaccard_sim = torch.tensor(jaccard_sim, device="cuda")
            # Convert the scalar to a 1D tensor
            #jaccard_sim = jaccard_sim.unsqueeze(0)
            #print("jaccard_sim of {} and {} is {}".format(i,j,jaccard_sim)) 
            #jaccard_similarity = torch.cat((jaccard_similarity, jaccard_sim), dim=0)
            del dst_neighbors
            #del jaccard_sim
            del dst_node
            #print("j loop end {}".format(j))
        del src_neighbors    
        #print("i loop end {}".format(i))
        
            # Compute intersection and union
            #intersection = len(src_neighbors & dst_neighbors)
            #union = len(src_neighbors | dst_neighbors)
        
            # Compute Jaccard similarity
                #if union > 0:
                #jaccard_similarity.append(intersection / union)
                #print("jaccard_similarity: ",jaccard_similarity)
            #else:
            #jaccard_similarity.append(0)  # Handle the case where there is no union

    # Convert the list to a tensor on GPU
    #jaccard_similarity_tensor = torch.tensor(jaccard_similarity, device="cuda")
    Jaccard_time_end = time.time()

    # The jaccard_similarity_tensor now contains the Jaccard similarity for each edge
    #print("jaccard_similarity_tensor: ",jaccard_similarity_tensor)
    print("jaccard_similarity: ",jaccard_similarity)
    print("Length of jaccard_similarity",len(jaccard_similarity))
    print("Jaccard_time: ",Jaccard_time_end - Jaccard_time_start)
    #weight_vector1 = cosine_similarities + jaccard_similarity
    weight_vector1 = torch.add(cosine_similarities,jaccard_similarity)
    #weight_vector1 = torch.add(weights,jaccard_similarity)
    weight_vector = weight_vector1 * 50
    weight_vector = torch.round(weight_vector)
    weight_vector = weight_vector.to(torch.int64)
    print("weight_vector",weight_vector)
    print("Length of weight_vector", len(weight_vector))
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
    print("Start Partitioning Weight_graph.....")
    start = time.time()
    node_parts_weight = dgl.metis_partition_assignment(G,nopart)
    # cut, membership = pymetis.part_graph(nopart, adjacency=None, xadj=xadj, adjncy=adjncy, vweights=None, eweights=adjwgt)
    # cut, membership = pymetis.part_graph(5, xadj=xadj, adjncy=adjncy, eweights=adjwgt)
    #RG = dgl.reorder_graph(G, node_permute_algo='metis', edge_permute_algo='dst', permute_config={'k':nopart})
    end = time.time()
    totalTime = totalTime + (end-start)
    print("Partition is Done !!!!!\t Time of Partition is :",round((end-start),4), "Seconds")
    mem_usage = (psutil.Process().memory_info().rss)/(1024 * 1024 * 1024)
    print(f"Current memory usage: { (mem_usage)} bytes")
    print("node_parts_weight: ",node_parts_weight)
    # print("cuts: ", cut)
    # print("membership: ", membership)


    # node_parts_weight = np.sort(node_parts_weight)
    # node_parts = np.sort(node_parts)

    # row_ptr=np.array(RG.adj_sparse('csr')[0])
    # col_idx=np.array(RG.adj_sparse('csr')[1])
    # row_ptr_s=len(row_ptr)
    # col_idx_s=len(col_idx)


    del g_row_ptr
    del g_col_idx
    del row_ptr
    del col_idx
    del g_weight_arr
    #del g_sum
    del weight_arr
    del weight_vector
    del weight_vector1
    #del sum
    #cp.cuda.runtime.free(intptr_t temp_arr)
    cp._default_memory_pool.free_all_blocks()
    end1 = time.time()
    print("Preprocess Successfull!!!! \tTime Taken of Prepr weight_vector1 = torch.add(cosine_similarities,jaccard_similarity)ocess is :",round((end1-start1),4), "Seconds")
