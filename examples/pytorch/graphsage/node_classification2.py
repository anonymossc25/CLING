import argparse
import time

import dgl
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
from dgl.data import CoraGraphDataset,RedditDataset,FlickrDataset
import numpy as np


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, cluster_id, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            cluster_id,
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


def layerwise_infer(device, graph, nid, cluster_id, model, num_classes, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(
            graph, device, cluster_id, batch_size
        )  # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
        return MF.accuracy(
            pred, label, task="multiclass", num_classes=num_classes
        )


def train(args, device, g, cluster_id,dataset, model, num_classes):
    # create sampler & dataloader
    #train_idx = dataset.train_idx.to(device)
    #val_idx = dataset.val_idx.to(device)
    train_mask=g.ndata['train_mask']
    val_mask=g.ndata['val_mask']
    train_idx = torch.nonzero(train_mask).squeeze().to(device)
    val_idx = torch.nonzero(val_mask).squeeze().to(device)
    sampler_time = time.time()

    sampler = NeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")],  # fanout for [layer-0, layer-1, layer-2]
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
        cluster_id,
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
        cluster_id,
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
    epoch_lines = []

    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        execution_time = 0.0
        start_time1 = time.time()
        #print("Epoch",epoch)
        iter_time = time.time()
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            #print("Iteration Time ",time.time() - iter_time)
            batch_start_time = time.time()
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            #print("batch Time", batch_time)
            iter_time = time.time()

        end_time1 = time.time()
        execution_time = end_time1 - start_time1
        total_training_time += execution_time    
        acc = evaluate(model, g, val_dataloader, num_classes)
        #print(
         #   "\nEpoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Time : {}\n".format(
          #       epoch, total_loss / (it + 1), acc.item(), execution_time
           #  )
        #)
        epoch_line = "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Time : {}".format(
                    epoch, total_loss / (it + 1), acc.item(), execution_time
        )
        epoch_lines.append(epoch_line)
    tt_time = "Total Training time {:.4f}".format( total_training_time)
    epoch_lines.append(tt_time)
    return epoch_lines

    #print( "Total Training time {:.4f}".format( total_training_time))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
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
        help="Dataset name ('cora', 'flickr', 'reddit','ogbn-products','ogbn-arxiv').",
    )
    parser.add_argument("--fan_out", type=str, default="10,10,10")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epoch", type=int, default=10)
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
    elif args.dataset == "ogbn-products":
        dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
    elif args.dataset == "ogbn-arxiv":
        dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv"))
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    g = dataset[0]
    #start_time = time.time()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    #print(f"Training in {args.mode} mode.")
    #part_array = get_part_array(g)
    test_mask=g.ndata['test_mask']
    test_idx = torch.nonzero(test_mask).squeeze()
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    columns = ['Data']
    file = pd.read_csv('/data/surendra/workspace/dgl_cluster/python/dgl/sampling/cluster/cluster_id.txt',names=columns)
    #print("This might a take while..")
    Data = file['Data']
    Data = np.array(Data)
    cluster_id = torch.from_numpy(Data)
    #cluster_id = dgl.ndarray.array(Data)
    #cluster_id = cluster_id.to("cuda" if args.mode == "puregpu" else "cpu")
    cluster_id = cluster_id.to(torch.device('cuda') if args.mode == "puregpu" else "cpu")
    #print(type(cluster_id))
    print("Device of cluster_id ", cluster_id.device)

    num_classes = dataset.num_classes
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, 256, out_size).to(device)

    # convert model and graph to bfloat16 if needed
    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        model = model.to(dtype=torch.bfloat16)

    # model training
    #print("Training...")
    epoch_lines = train(args, device, g, cluster_id, dataset, model, num_classes)

    # test the model
    #print("Testing...")
    acc = layerwise_infer(
        device, g, test_idx, cluster_id, model, num_classes, batch_size=4096
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

