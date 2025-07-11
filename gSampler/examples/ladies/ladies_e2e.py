import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.utils import gather_pinned_tensor_rows
from dgl.nn.pytorch import SAGEConv
import gs
from gs.utils import SeedGenerator, load_reddit, load_ogb, create_block_from_coo
import numpy as np
import time
import tqdm
import argparse
from typing import List


def normalized_laplacian_edata(g, weight=None):
    with g.local_scope():
        if weight is None:
            weight = "W"
            g.edata[weight] = torch.ones(g.number_of_edges(), device=g.device)
        g_rev = dgl.reverse(g, copy_edata=True)
        g.update_all(fn.copy_e(weight, weight), fn.sum(weight, "v"))
        g_rev.update_all(fn.copy_e(weight, weight), fn.sum(weight, "u"))
        g.ndata["u"] = g_rev.ndata["u"]
        g.apply_edges(lambda edges: {"w": edges.data[weight] / torch.sqrt(edges.src["u"] * edges.dst["v"])})
        return g.edata["w"]


class DGLSAGEModel(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_size, hid_size, "mean"))
        for i in range(num_layers - 2):
            self.layers.append(SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)

    def forward(self, blocks, h):
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


def ladies_sampler(A: gs.BatchMatrix, fanouts: List, seeds: torch.Tensor, seeds_ptr: torch.Tensor):
    ret = []
    for K in fanouts:
        subA = A[:, seeds::seeds_ptr]
        subA.edata["p"] = subA.edata["w"] ** 2
        prob = subA.sum("p", axis=1)
        neighbors, probs_ptr = subA.all_rows()
        sampleA, select_index = subA.collective_sampling(K, prob, probs_ptr, False)
        sampleA = sampleA.div("w", prob[select_index], axis=1)
        out = sampleA.sum("w", axis=0)
        sampleA = sampleA.div("w", out, axis=0)
        seeds, seeds_ptr = sampleA.all_nodes()
        ret.append(sampleA.to_dgl_block())
    return ret


def compute_acc(pred, label):
    return (pred.argmax(1) == label).float().mean()


def train(dataset, args):
    device = args.device
    use_uva = args.use_uva
    fanouts = [int(x.strip()) for x in args.samples.split(",")]

    g, features, labels, n_classes, splitted_idx = dataset
    train_nid, val_nid = splitted_idx["train"], splitted_idx["valid"]
    g = g.to(device)
    weight = normalized_laplacian_edata(g)
    train_nid, val_nid = train_nid.to(device), val_nid.to(device)
    features, labels = features.to(device), labels.to(device)
    csc_indptr, csc_indices, edge_ids = g.adj_sparse("csc")
    weight = weight[edge_ids].to(device)
    if use_uva and device == "cpu":
        features, labels = features.pin_memory(), labels.pin_memory()
        csc_indptr = csc_indptr.pin_memory()
        csc_indices = csc_indices.pin_memory()
        train_nid, val_nid = train_nid.pin_memory(), val_nid.pin_memory()
    m = gs.Matrix()
    m.load_graph("CSC", [csc_indptr, csc_indices])
    m.edata["w"] = weight
    bm = gs.BatchMatrix()
    bm.load_from_matrix(m)

    batch_size = args.batchsize
    # batch_size = 12800
    small_batch_size = args.batchsize
    num_batches = int((batch_size + small_batch_size - 1) / small_batch_size)
    orig_seeds_ptr = torch.arange(num_batches + 1, dtype=torch.int64, device="cuda") * small_batch_size
    print(batch_size, small_batch_size, fanouts)

    train_seedloader = SeedGenerator(train_nid, batch_size=batch_size, shuffle=True, drop_last=False)
    val_seedloader = SeedGenerator(val_nid, batch_size=batch_size, shuffle=True, drop_last=False)
    model = DGLSAGEModel(features.shape[1], 256, n_classes, len(fanouts)).to("cuda")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    compile_func = gs.jit.compile(func=ladies_sampler, args=(bm, fanouts, train_seedloader.data[:batch_size], orig_seeds_ptr))

    torch.cuda.synchronize()
    static_memory = torch.cuda.memory_allocated()
    print("memory allocated before training:", static_memory / (1024 * 1024 * 1024), "GB")

    epoch_time = []
    cur_time = []
    acc_list = []
    start = time.time()
    epoch_lines = []
    for epoch in range(args.num_epoch):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        tic = time.time()
        model.train()
        num_batches = int((batch_size + small_batch_size - 1) / small_batch_size)
        #for it, seeds in enumerate(tqdm.tqdm(train_seedloader)):
        for it, seeds in enumerate(train_seedloader):
            seeds_ptr = orig_seeds_ptr
            if it == len(train_seedloader) - 1:
                num_batches = int((seeds.numel() + small_batch_size - 1) / small_batch_size)
                seeds_ptr = torch.arange(num_batches + 1, dtype=torch.int64, device="cuda") * small_batch_size
                seeds_ptr[-1] = seeds.numel()
            all_blocks = compile_func(bm, fanouts, seeds, seeds_ptr)

            for rank in range(num_batches):
                blocks = []
                for layer in range(len(fanouts)):
                    blocks.insert(0, all_blocks[layer][rank])
                input_nodes = blocks[0].srcdata["_ID"]
                output_nodes = seeds[seeds_ptr[rank] : seeds_ptr[rank + 1]].cuda()
                #print("layer1:", blocks[0])
                #print("layer2:", blocks[1])
                #print("layer3:", blocks[2])
                if use_uva:
                    batch_inputs = gather_pinned_tensor_rows(features, input_nodes)
                    batch_labels = gather_pinned_tensor_rows(labels, output_nodes)
                else:
                    batch_inputs = features[input_nodes].to("cuda")
                    batch_labels = labels[output_nodes].to("cuda")

                batch_pred = model(blocks, batch_inputs)
                is_labeled = batch_labels == batch_labels
                batch_labels = batch_labels[is_labeled].long()
                batch_pred = batch_pred[is_labeled]
                loss = F.cross_entropy(batch_pred, batch_labels)
                opt.zero_grad()
                loss.backward()
                opt.step()

        model.eval()
        val_pred = []
        val_labels = []
        with torch.no_grad():
            num_batches = int((batch_size + small_batch_size - 1) / small_batch_size)
            #for it, seeds in enumerate(tqdm.tqdm(val_seedloader)):
            for it, seeds in enumerate(val_seedloader):
                seeds_ptr = orig_seeds_ptr
                if it == len(val_seedloader) - 1:
                    num_batches = int((seeds.numel() + small_batch_size - 1) / small_batch_size)
                    seeds_ptr = torch.arange(num_batches + 1, dtype=torch.int64, device="cuda") * small_batch_size
                    seeds_ptr[-1] = seeds.numel()
                all_blocks = compile_func(bm, fanouts, seeds, seeds_ptr)

                for rank in range(num_batches):
                    blocks = []
                    for layer in range(len(fanouts)):
                        blocks.insert(0, all_blocks[layer][rank])
                    input_nodes = blocks[0].srcdata["_ID"]
                    output_nodes = seeds[seeds_ptr[rank] : seeds_ptr[rank + 1]].cuda()
                    if use_uva:
                        batch_inputs = gather_pinned_tensor_rows(features, input_nodes)
                        batch_labels = gather_pinned_tensor_rows(labels, output_nodes)
                    else:
                        batch_inputs = features[input_nodes].to("cuda")
                        batch_labels = labels[output_nodes].to("cuda")

                    batch_pred = model(blocks, batch_inputs)
                    is_labeled = batch_labels == batch_labels
                    batch_labels = batch_labels[is_labeled].long()
                    batch_pred = batch_pred[is_labeled]
                    val_pred.append(batch_pred)
                    val_labels.append(batch_labels)

        acc = compute_acc(torch.cat(val_pred, 0), torch.cat(val_labels, 0)).item()
        acc_list.append(acc)

        torch.cuda.synchronize()
        end = time.time()
        cur_time.append(end - start)
        epoch_time.append(end - tic)

        #print("Epoch {:05d} | Val Acc {:.4f} | E2E Time {:.4f} s | Accumulated Time {:.4f} s".format(epoch, acc, epoch_time[-1], cur_time[-1]))
        epoch_line = "Epoch {:05d} | Val Acc {:.4f} | E2E Time {:.4f} s | Accumulated Time {:.4f} s".format(epoch, acc, epoch_time[-1], cur_time[-1])
        epoch_lines.append(epoch_line)

    torch.cuda.synchronize()
    total_time = time.time() - start

    #print("Total Elapse Time:", total_time)
    #Total_time = "Total Elapse Time:", Total_time
    Total_time = "Total time {:.4f}".format(total_time)
    epoch_lines.append(Total_time)

    #print("Average Epoch Time:", np.mean(epoch_time[3:]))
    with open('epoch_data.txt', 'w') as file:
        for value in epoch_lines:
            file.write(str(value) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Training model on gpu or cpu",
    )
    parser.add_argument(
        "--use-uva",
        type=bool,
        default=False,
        help="Wether to use UVA to sample graph and load feature",
    )
    parser.add_argument(
        "--dataset",
        default="ogbn-products",
        choices=["reddit", "ogbn-products", "ogbn-arxiv", "ogbn-papers100m"],
        help="which dataset to load for training",
    )
    parser.add_argument("--batchsize", type=int, default=512, help="batch size for training")
    parser.add_argument("--samples", default="4000,4000,4000", help="sample size for each layer")
    parser.add_argument("--num-epoch", type=int, default=100, help="numbers of epoch in training")
    args = parser.parse_args()
    print(args)

    if args.dataset == "reddit":
        dataset = load_reddit()
    elif args.dataset == "ogbn-products":
        dataset = load_ogb("ogbn-products", "../dataset")
    elif args.dataset == "ogbn-arxiv":
        dataset = load_ogb("ogbn-arxiv", "../dataset")
    
    elif args.dataset == "ogbn-papers100m":
        dataset = load_ogb("ogbn-papers100M", "../dataset")
    print(dataset[0])
    train(dataset, args)
