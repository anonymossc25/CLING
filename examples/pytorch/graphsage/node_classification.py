import argparse
import numpy as np
import dgl
import time
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset


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

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
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


def train(args, device, g, dataset, model, num_classes, part_array):
    # create sampler & dataloader
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    execution_time = 0.0
    start_time = time.time()
    sampler = NeighborSampler(
        [30, 30, 30],  # fanout for [layer-0, layer-1, layer-2]
        # part_array,
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    end_time = time.time()
    execution_time = end_time - start_time
    # G = dgl.to_homogeneous(g)
    # print(G, type(G))
    # print("total sampling time:", execution_time, "seconds")

    # _computed_array = dgl.metis_partition_assignment(G, 4, balance_ntypes=None, balance_edges=False, mode='k-way', objtype='cut')
    # dgl.distributed.partition_graph(G, 'test', 2, out_path='output/')

    use_uva = args.mode == "mixed"
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        # part_array,
        device=device,
        batch_size= int(args.batch_size),
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        # part_array,
        device=device,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(1):
        model.train()    
        total_loss = 0
        execution_time = 0.0
        start_time1 = time.time()
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        end_time1 = time.time()
        execution_time = end_time1 - start_time1
        # print("training time:", execution_time, "seconds")
        acc = evaluate(model, g, val_dataloader, num_classes)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Time : {}".format(
                 epoch, total_loss / (it + 1), acc.item(), execution_time
             )
         )
        

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
        default="ogbn-products",
        choices=["ogbn-products", "ogbn-arxiv", "ogbn-papers100M", "reddit"],
        help="pass dataset",
    )
    parser.add_argument(
        "--batch_size",
        default="1024",
        choices=["1024", "2048", "4096", "8192"],
        help="batch_size for train",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    # load and preprocess dataset
    print("Loading data")
    dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset))
    g = dataset[0]
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    num_classes = dataset.num_classes
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    # create GraphSAGE model)
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, 256, out_size).to(device)

    # convert model and graph to bfloat16 if needed
    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        model = model.to(dtype=torch.bfloat16)

    # out partion create array 
    part_array = np.ones(5)
    # part_array = torch.from_numpy(part_array)
    # model training
    print("Training...")
    execution_time1 = 0.0
    start_time1 = time.time()
    train(args, device, g, dataset, model, num_classes, part_array)
    end_time1 = time.time()
    execution_time1 = end_time1 - start_time1
    # print("total training time:", execution_time1, "seconds")


    # test the model
    print("Testing...")
    acc = layerwise_infer(
        device, g, dataset.test_idx, model, num_classes, batch_size=4096
    )
    print("Test Accuracy {:.4f}".format(acc.item()))
