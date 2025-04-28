import dgl
import argparse
import time
import matplotlib.pyplot as plt
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from dgl.sampling.metis_sampling import *
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
#import dgl.data.CoraGraphDataset
from dgl.data import CoraGraphDataset,RedditDataset,FlickrDataset,YelpDataset
import numpy as np
import cupy as cp


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "gcn"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "gcn"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "gcn"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x

        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            #print("Checking weight", block.edata['weight'])
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            #cluster_id,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


def evaluate(model, graph, dataloader, num_classes):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


def layerwise_infer(device, graph, nid, model, num_classes, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(
            graph, device, batch_size
        )  # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
        return MF.accuracy(
            pred, label, task="multiclass", num_classes=num_classes
        )


def train(args, device, g, 
          # cluster_id,
          dataset, model, num_classes):

    # create sampler & dataloader
    #train_idx = dataset.train_idx.to(device)
    #val_idx = dataset.val_idx.to(device)
    train_mask=g.ndata['train_mask']
    val_mask=g.ndata['val_mask']
    train_idx = torch.nonzero(train_mask).squeeze().to(device)
    #print("train :",len(train_idx))
    val_idx = torch.nonzero(val_mask).squeeze().to(device)
    #print("val: ",len(val_idx))
    sampler_time = time.time()

    sampler = NeighborSampler(
        [int(fanout) for fanout in args.fanout.split(",")],  # fanout for [layer-0, layer-1, layer-2]
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    sampler_end_time = time.time()
    sampler_time = sampler_end_time-sampler_time
    #print("Sampler time:", sampler_time, "seconds")


    use_uva = args.mode == "mixed"
    Tdataload_time = time.time()
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        #cluster_id,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )
    #print("After train_dataloader")
    Tdataload_end_time = time.time()
    Tdataload_time = Tdataload_end_time-Tdataload_time
    #print("Tdataload time:", Tdataload_time, "seconds")
    Vdataload_time = time.time()

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        #cluster_id,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )
    Vdataload_end_time = time.time()
    Vdataload_time = Vdataload_end_time-Vdataload_time
    #print("Vdataload time:", Vdataload_time, "seconds")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    total_training_time = 0.0
    total_for_loop_time = 0.0
    total_model_time = 0.0
    epoch_lines = []

    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        execution_time = 0.0
        iteration_time1 =0.0
        model_time1 =0.0
        x_y_time = 0.0
        pred_time = 0.0
        loss_time = 0.0
        backward_time = 0.0
        optim_time = 0.0
        start_time1 = time.time()
        #print("Epoch",epoch)
        iter_time = time.time()
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            iteration_time = time.time() - iter_time
            #print("Iteration Time ",iteration_time)
            batch_start_time = time.time()
            #print("Batch processing");

            start_x_y_time = time.time()
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            #print("1st layer:",blocks[0])
            #print("2nd layer:",blocks[1])
            #print("3rd layer:",blocks[-1])
            end_x_y_time = time.time()

            start_pred_time = time.time()
            #print("before forward pass\n");
            y_hat = model(blocks, x)
            end_pred_time = time.time()

            start_loss_time = time.time()
            #print("After forward pass\n");
            loss = F.cross_entropy(y_hat, y)
            end_loss_time = time.time()

            start_backward_time = time.time()
            opt.zero_grad()
            #print("before backward pass\n");
            loss.backward()
            end_backward_time = time.time()

            start_optim_time = time.time()
            #print("after backward pass\n");
            opt.step()
            end_optim_time = time.time()

            total_loss += loss.item()
            batch_end_time = time.time()
            model_time = batch_end_time - batch_start_time
            #print("model Time ", model_time)
            iteration_time1 += iteration_time
            model_time1 += model_time
            x_y_time1 = end_x_y_time - start_x_y_time
            x_y_time += x_y_time1
            pred_time1 = end_pred_time - start_pred_time
            pred_time += pred_time1
            backward_time1 = end_backward_time - start_backward_time
            backward_time += backward_time1
            optim_time1 = end_optim_time - start_optim_time
            optim_time += optim_time1
            #minibatch_time = time.time()-iter_time
            #print("mini batch time ",minibatch_time)
            #print("loop time {:.5f} | model time {:.5f} | xy time {:.5f} | forward time {:.5f} | backword time {:.5f} | optim time {:.5f}".format(iteration_time, model_time, x_y_time1, pred_time1, backward_time1, optim_time1))
            iter_time = time.time()
            #batch_start_time = time.time()

        end_time1 = time.time()
        execution_time = end_time1 - start_time1
        total_training_time += execution_time
        total_for_loop_time += iteration_time1
        total_model_time += model_time1
        acc = evaluate(model, g, val_dataloader, num_classes)
        #print(
         #   "\nEpoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Time : {}\n".format(
          #       epoch, total_loss / (it + 1), acc.item(), execution_time
           #  )
        #)
        #epoch_line = "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Time : {}".format(epoch, total_loss / (it + 1), acc.item(), execution_time )
        epoch_line = "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Time : {:.4f} | Loop_Time : {:.4f} | Model_Time : {:.4f} | x_y_time : {:.4f} | pred_time : {:.4f} | backward_time : {:.4f} | optim_time : {:.4f} ".format(epoch, total_loss / (it + 1), acc.item(), execution_time, iteration_time1, model_time1, x_y_time, pred_time, backward_time, optim_time )
        #print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Time : {:.4f} | Loop_Time : {:.4f} | Model_Time : {:.4f} | x_y_time : {:.4f} | pred_time : {:.4f} | backward_time : {:.4f} | optim_time : {:.4f} ".format(epoch, total_loss / (it + 1), acc.item(), execution_time, iteration_time1, model_time1, x_y_time, pred_time, backward_time, optim_time ))

        epoch_lines.append(epoch_line)
    #tt_str = "total for loop time, total model time, total_training_time"
    #tt_time = "{:.4f}, {:.4f}, {:.4f}".format(total_for_loop_time, total_model_time, total_training_time)
    tt_time = "Sampling time: {:.4f}, Model training time: {:.4f}, Total time {:.4f}".format(total_for_loop_time, total_model_time, total_training_time)
    #epoch_lines.append(tt_str)
    epoch_lines.append(tt_time)     
    #tt_time = "Total Training time {:.4f}".format( total_training_time)
    #epoch_lines.append(tt_time)
    return epoch_lines

    #print( "Total Training time {:.4f}".format( total_training_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="puregpu",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    parser.add_argument(
        "--method",
        default="cling",
        choices=["graphsage", "cling"],
        help="graphsagse vs cling",
        )
    parser.add_argument(
        "--dt",
        type=str,
        default="float",
        help="data type(float, bfloat16)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-arxiv",
        #help="Dataset name ('cora', 'flickr', 'reddit', 'yelp', 'ogbn-products','ogbn-arxiv').",
    )
    parser.add_argument("--fanout", type=str, default="10,10,10")
    parser.add_argument("--num_clusters", type=str, default="20")
    #parser.add_argument("--fan_out", type=str, default="10,10,10,10,10")
    #parser.add_argument("--fan_out", type=str, default="25,10")

    #parser.add_argument("--fan_out", type=str, default="15,15,15")

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epoch", type=int, default=100)
    args = parser.parse_args()
    #print(f"Training with DGL built-in GraphConv module.")

    # load and preprocess dataset
    if args.dataset == "cora":
        dataset = CoraGraphDataset()
    elif args.dataset == "citeseer":
        dataset = CiteseerGraphDataset()
    elif args.dataset == "pubmed":
        dataset = PubmedGraphDataset()
    elif args.dataset == "wisconsin":
        dataset = WisconsinDataset()
    elif args.dataset == "flickr":
        dataset = FlickrDataset()
    elif args.dataset == "reddit":
        dataset = RedditDataset()
    elif args.dataset == "yelp":
        dataset = YelpDataset()
    elif args.dataset == "ogbn-products":
        dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
    elif args.dataset == "ogbn-arxiv":
        dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv"))
    elif args.dataset == "amazon_products":
        load_path = '/data/Dataset/gnn_dataset/amazon_products.dgl'
        dataset, _ = dgl.load_graphs(load_path)
    elif args.dataset == "cit-net":
        load_path = '/data/Dataset/gnn_dataset/citations_network_graph.dgl'
        dataset, _ = dgl.load_graphs(load_path)    
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    g = dataset[0]
    
    """
    #printing and ploting graph degree related information.
    out_degrees = np.array(g.out_degrees())
    max_value = np.max(out_degrees)
    avg_value = np.mean(out_degrees)
    print("maximum degree : ",max_value)
    print("Average degree : ",avg_value)
    # Count the number of nodes with in-degree less than 100
    num_nodes_less_than_100 = len(out_degrees[out_degrees < 100])
    num_nodes_less_than_128 = len(out_degrees[out_degrees < 64])
    print("Total number of nodes with in-degree less than 100:", num_nodes_less_than_100)
    print("Total number of nodes with in-degree less than 64:", num_nodes_less_than_128)
    unique_values, frequencies = np.unique(out_degrees, return_counts=True)
    
    # Create a TSV file
    output_file = str(args.dataset) + ".tsv"
    # Write unique values and frequencies to the TSV file
    np.savetxt(output_file, np.column_stack((unique_values, frequencies)), delimiter='\t', fmt='%d')

    plt.bar(unique_values, frequencies)
    plt.xlabel('Degree of Vertex', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    #plt.ylim(0, 200000)
    #max_y = max(frequencies)
    highest_y = np.max(frequencies)
    highest_x = unique_values[np.argmax(frequencies)]
    plt.annotate(str(highest_y), xy=(highest_x, highest_y), ha='center', va='bottom', fontsize=18)

    # Add text annotation for the highest value
    #plt.text(unique_values[frequencies.index(max_y)], max_y, str(max_y), ha='center', va='bottom')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.yscale('log')
    plt.ylim(1, 10**7)
    #plt.title('Degree Distribution')
    plot_name = str(args.dataset) + ".eps"
    plt.savefig(plot_name, format='eps')
    """
    #start_time = time.time()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    #print(f"Training in {args.mode} mode.")
    #[str(fanout) for fanout in args.fan_out.split(",")],  # fanout for [layer-0, layer-1, layer-2]
    #print(args.num_clusters)
    #fanout = args.fan_out.split(",")[0]
    #print(fanout)
    #print(type(fanout))
    if args.method == "graphsage":
        method = 0
    elif args.method == "cling":
        method = 1
    else:
        print("please provide proper method, 0 for graphsage 1 for cling")

    part_array = get_part_array(g, args.dataset, args.num_clusters)
    node_array = get_representative_array(g, args.dataset, args.num_clusters)
    method = get_method(method)
    test_mask=g.ndata['test_mask']
    test_idx = torch.nonzero(test_mask).squeeze()
    #print("test: ",len(test_idx))
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    #columns = ['Data']
    #file = pd.read_csv('/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster/cluster_id.txt',names=columns)
    #print("This might a take while..")
    #Data = file['Data']
    #Data = np.array(Data)
    #cluster_id = torch.from_numpy(Data)
    #cluster_id = dgl.ndarray.array(Data)
    #cluster_id = cluster_id.to("cuda" if args.mode == "puregpu" else "cpu")
    # cluster_id = cluster_id.to(torch.device('cuda') if args.mode == "puregpu" else "cpu")
    #print(type(cluster_id))
    #print("Device of cluster_id ", cluster_id.device)

    num_classes = dataset.num_classes
    #num_classes = 107
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    #out_size = 107
    model = SAGE(in_size, 256, out_size).to(device)

    # convert model and graph to bfloat16 if needed
    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        model = model.to(dtype=torch.bfloat16)

    # model training
    #print("Training...")
    epoch_lines = train(args, device, g, 
                        # cluster_id, 
                        dataset, model, num_classes)

    # test the model
    print("Testing...")
    acc = layerwise_infer(
        device, g, test_idx, model, num_classes, batch_size=4096
    )
    #acc = layerwise_infer(
        #device, g, dataset.test_idx, model, num_classes, batch_size=4096
    #)
    #end_time = time.time()
    #execution_time = end_time - start_time
    #print("Test Accuracy {:.4f}".format(acc.item()))
    #print("Execution time:", execution_time, "seconds")
    Accuracy = "Test Accuracy {:.4f}".format(acc.item())
    #tt_time = "Total Training time {:.4f}".format( total_training_time)
    #epoch_lines.append(tt_time)
    epoch_lines.append(Accuracy)
    with open('epoch_data.txt', 'w') as file:
        for value in epoch_lines:
            file.write(str(value) + '\n')

