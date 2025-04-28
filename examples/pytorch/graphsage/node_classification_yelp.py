#!/data/kishan/anaconda3/envs/dgl-dev-gpu-117/bin/python3
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
from dgl.metis_sampling import *
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset

from dgl.data import CoraGraphDataset,RedditDataset,FlickrDataset, YelpDataset


print("at the top")

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
    # return MF.accuracy(
    #     # torch.cat(y_hats),
    #     # torch.cat(ys),
    #     # task="multiclass",
    #     # num_classes=num_classes,
    #     (torch.cat(y_hats).sigmoid() > 0.5).int(),
    #     torch.cat(ys).int(),
    #     task="multiclass",
    #     num_classes=num_classes,
    # )
    preds = (torch.cat(y_hats).sigmoid() > 0.5).int()
    labels = torch.cat(ys).int()
    acc = (preds == labels).float().mean()  # mean over all labels and samples
    return acc
    # return acc.item()

def layerwise_infer(device, graph, nid, model, num_classes, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(
            graph, device, batch_size
        )  # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
                # Apply sigmoid to logits and threshold
        pred_labels = (pred.sigmoid() > 0.5).int()
        label = label.int()

        # Compute accuracy as exact match or elementwise
        correct = (pred_labels == label).float()
        acc = correct.mean()  # mean over all samples and classes
        return acc
        # return MF.accuracy(
        #     pred, label, task="multiclass", num_classes=num_classes
        # )


def train(args, device, g, dataset, model, num_classes):
    # create sampler & dataloader
    #train_idx = dataset.train_idx.to(device)
    #val_idx = dataset.val_idx.to(device)
    train_mask=g.ndata['train_mask']
    val_mask=g.ndata['val_mask']
    train_idx = torch.nonzero(train_mask).squeeze().to(device)
    val_idx = torch.nonzero(val_mask).squeeze().to(device)

    execution_time = 0.0
    start_time = time.time()
    sampler = NeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")],
        #[30, 30, 30],  # fanout for [layer-0, layer-1, layer-2]
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

    total_training_time = 0.0
    total_for_loop_time = 0.0
    total_model_time = 0.0
    total_loss_opt_time = 0.0
    epoch_lines = []
    for epoch in range(int(args.epoch)):
        model.train()    
        total_loss = 0
        execution_time = 0.0
        model_exe_time = 0.0
        loop_exe_time = 0.0
        x_y_time = 0.0
        pred_time = 0.0
        loss_opt_time = 0.0
        start_time1 = time.time()
        start_loop_time = time.time()
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            # print("blocks: ",blocks)
            # print("input nodes: ",input_nodes)
            # print("output nodes: ",output_nodes)
            end_loop_time = time.time()
            start_model_time = time.time()
            start_x_y_time = time.time()
            x = blocks[0].srcdata["feat"]
            # print("X shape",blocks[0])
            # print("X:",x)
            y = blocks[-1].dstdata["label"]
            # print("dst nodes of block -1: ",blocks[-1].dstdata)
            end_x_y_time = time.time()
            # print("Y shape", blocks[-1])
            # print("Y: ",y)

            start_pred_time = time.time()
            y_hat = model(blocks, x)
            # print(type(y_hat))
            # print("y_hat: ", y_hat)
            # print("len: ", len(y_hat))
            end_pred_time = time.time()
            
            start_loss_time = time.time()
            # y = y.float()
            y = y.long()
            # print("y :", y)
            # print("type: ", type(y))
            # print("len: ", len(y))
            # print("y_hat type:", y_hat.dtype)
            # print("y type:", y.dtype)
            # print("y_hat shape:", y_hat.shape)
            # print("y shape:", y.shape)
            #
            # y = torch.argmax(y, dim=1)

            # loss = F.cross_entropy(y_hat, y)
            # loss = F.binary_cross_entropy_with_logits(y_hat, y.flot())
            loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            end_loss_time = time.time()
            end_model_time = time.time()

            model_exe_time += end_model_time - start_model_time
            loop_exe_time += end_loop_time - start_loop_time
            x_y_time += end_x_y_time - start_x_y_time
            pred_time += end_pred_time - start_pred_time
            loss_opt_time =+ end_loss_time - start_loss_time

            start_loop_time = time.time()

        end_time1 = time.time()
        execution_time = end_time1 - start_time1
        total_training_time += execution_time
        total_for_loop_time += loop_exe_time
        total_model_time += model_exe_time
        total_loss_opt_time += loss_opt_time
        # print("training time:", execution_time, "seconds")
        acc = evaluate(model, g, val_dataloader, num_classes)
        # print(
        #     "\nEpoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Time : {}".format(
        #          epoch, total_loss / (it + 1), acc.item(), execution_time
        #      )
        #  )
        epoch_line = "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Time : {:.4f} | loop {:.4f} | Model {:.4f} | x_y_time {:.4f} | pred {:.4f} | loss_time {:.4f}".format(
                    epoch, total_loss / (it + 1), acc.item(), execution_time, loop_exe_time, model_exe_time, x_y_time, pred_time, loss_opt_time
        )
        epoch_lines.append(epoch_line)
    tt_str = "total for loop time, total model time, total loss time, total_training_time"
    tt_time = "{:.4f}, {:.4f}, {:.4f}, {:.4f}".format(total_for_loop_time, total_model_time, total_loss_opt_time, total_training_time)
    epoch_lines.append(tt_str)
    epoch_lines.append(tt_time)
    return epoch_lines

    # print("\nTotal Training Time {:.4f} seconds".format(total_training_time))
        

if __name__ == "__main__":

    print("inside the main")
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
        # choices=["ogbn-products", "ogbn-arxiv", "ogbn-papers100M", "reddit"],
        help="pass dataset",
    )
    parser.add_argument(
        "--batch_size",
        default="1024",
        # choices=["1024", "2048", "4096", "8192"],
        help="batch_size for train",
    )
    parser.add_argument(
        "--epoch",
        default="1",
        help="batch_size for train",
    )
    parser.add_argument(
        "--method",
        type=str,
        default = None,
        choices=["metis", "rm", "contig"],
        help="Partition method for sampling"
    )
    parser.add_argument("--fan_out", type=str, default="10,10,10")
    parser.add_argument("--parts", type=int, default=10)
    parser.add_argument("--spmm", default="cusparse")
    parser.add_argument("--sampling", default="default")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"\nTraining in {args.mode} mode.")

    # load and preprocess dataset
    # print("\nLoading data")
    # dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset))
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
    else:
        dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset))
        # raise ValueError("Unknown dataset: {}".format(args.dataset))

    if args.spmm == "cusparse":
        spmm_method = 0
    elif args.spmm == "respmm":
        spmm_method = 1
    elif args.spmm == "gespmm":
        spmm_method = 2
    else:
        print("please provide valid spmm mathod like respmm or gespmm. default value is cusparse")
        
    if args.sampling == "metis":
        sampling_method = 0
    elif args.sampling == "default":
        sampling_method = 1
    else:
        print("please provide valid sampling mathod like metis (0) or default (1). default value is cusparse")

    g = dataset[0]
    print("metis partition called")
    part_array = get_part_array(g, args.parts, args.method, spmm_method, sampling_method)
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")
    test_mask=g.ndata['test_mask']
    test_idx = torch.nonzero(test_mask).squeeze().to("cpu")

    num_classes = dataset.num_classes

    # create GraphSAGE model)
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, 256, out_size).to(device)

    # convert model and graph to bfloat16 if needed
    if args.dt == "bfloat16":
        g = dgl.to_bfloat16(g)
        model = model.to(dtype=torch.bfloat16)

    # out partion create array 
    #part_array = np.ones(5)

    # part_array = torch.from_numpy(part_array)
    # model training
    # print("\nTraining...")
    execution_time1 = 0.0
    start_time1 = time.time()
    #train(args, device, g, dataset, model, num_classes)
    epoch_lines=train(args, device, g, dataset, model, num_classes)
    end_time1 = time.time()
    execution_time1 = end_time1 - start_time1
    # print("total training time:", execution_time1, "seconds")


    # test the model
    print("\nTesting...")
    acc = layerwise_infer(
        device, g, test_idx, model, num_classes, batch_size=8192
    )
    #print("\nTest Accuracy {:.4f}".format(acc.item()))
    Accuracy = "Test Accuracy {:.4f}".format(acc.item())
    epoch_lines.append(Accuracy)
    with open('epoch_data.txt', 'w') as file:
        for value in epoch_lines:
            file.write(str(value) + '\n')



