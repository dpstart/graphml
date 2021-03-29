import numpy as np
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import CoraGraphDataset

from model import SAGE


def load_cora():
    data = CoraGraphDataset()
    g = data[0]
    g.ndata["features"] = g.ndata["feat"]
    g.ndata["labels"] = g.ndata["label"]
    return g, data.num_classes


def inductive_split(g):
    """ Split graph into training, validation and test graph with masks"""
    train_g = g.subgraph(g.ndata["train_mask"])
    val_g = g.subgraph(g.ndata["train_mask"] | g.ndata["val_mask"])
    test_g = g

    return train_g, val_g, test_g


def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, nfeat, labels, val_nid, device, args):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with torch.no_grad():
        pred = model.inference(g, nfeat, device, args)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid].to(pred.device))


def run(args, device, data):

    (
        n_classes,
        train_g,
        val_g,
        test_g,
        train_nfeat,
        train_labels,
        val_nfeat,
        val_labels,
        test_nfeat,
        test_labels,
    ) = data

    in_feats = train_nfeat.shape[1]
    train_nid = torch.nonzero(train_g.ndata["train_mask"], as_tuple=True)[0]
    val_nid = torch.nonzero(val_g.ndata["val_mask"], as_tuple=True)[0]
    test_nid = torch.nonzero(test_g.ndata["train_mask"], as_tuple=True)[0]

    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")]
    )
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    model = SAGE(
        in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout
    )
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    avg = 0
    iter_output = []

    for epoch in range(args.num_epochs):

        tic = time.time()

        tic_step = time.time()

        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):

            batch_inputs, batch_labels = load_subtensor(
                train_nfeat, train_labels, seeds, input_nodes, device
            )
            blocks = [block.int().to(device) for block in blocks]

            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_output.append(len(seeds) / (time.time() - tic_step))

            if step % args.log_every == 0:

                acc = compute_acc(batch_pred, batch_labels)
                print(
                    f"Epoch {epoch} | Step {step} | Loss {loss.item()} | Train Acc {acc.item()} | Speed (samples/sec_{np.mean(iter_output[3:])}"
                )
                tic_step = time.time()

            toc = time.time()
            print("Epoch Time(s): {:.4f}".format(toc - tic))
            if epoch >= 5:
                avg += toc - tic
            if epoch % args.eval_every == 0 and epoch != 0:
                eval_acc = evaluate(
                    model, val_g, val_nfeat, val_labels, val_nid, device, args
                )
                print("Eval Acc {:.4f}".format(eval_acc))
                test_acc = evaluate(
                    model, test_g, test_nfeat, test_labels, test_nid, device,
                    args
                )
                print("Test Acc: {:.4f}".format(test_acc))

    print("Avg epoch time: {}".format(avg / (epoch - 4)))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument(
        "--gpu", type=int, default=0, help="GPU device ID. Use -1 for CPU training"
    )
    argparser.add_argument("--dataset", type=str, default="reddit")
    argparser.add_argument("--num-epochs", type=int, default=20)
    argparser.add_argument("--num-hidden", type=int, default=16)
    argparser.add_argument("--num-layers", type=int, default=2)
    argparser.add_argument("--fan-out", type=str, default="10,25")
    argparser.add_argument("--batch-size", type=int, default=1000)
    argparser.add_argument("--log-every", type=int, default=20)
    argparser.add_argument("--eval-every", type=int, default=5)
    argparser.add_argument("--lr", type=float, default=0.003)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of sampling processes. Use 0 for no extra process.",
    )
    argparser.add_argument(
        "--inductive", action="store_true", help="Inductive learning setting"
    )
    argparser.add_argument(
        "--data-cpu",
        action="store_true",
        help="By default the script puts all node features and labels "
        "on GPU when using it to save time for data copy. This may "
        "be undesired if they cannot fit in GPU memory at once. "
        "This flag disables that.",
    )
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = torch.device("cuda:%d" % args.gpu)
    else:
        device = torch.device("cpu")

    g, n_classes = load_cora()

    if args.inductive:

        train_g, val_g, test_g = inductive_split(g)
        train_nfeat = train_g.ndata.pop("features")
        val_nfeat = val_g.ndata.pop("features")
        test_nfeat = test_g.ndata.pop("features")

        train_labels = train_g.ndata.pop("labels")
        val_labels = val_g.ndata.pop("labels")
        test_labels = test_g.ndata.pop("labels")
    else:
        train_g = val_g = test_g = g
        train_nfeat = val_nfeat = test_nfeat = g.ndata.pop("features")
        train_labels = val_labels = test_labels = g.ndata.pop("labels")

    train_nfeat = train_nfeat.to(device)
    train_labels = train_labels.to(device)

    data = (
        n_classes,
        train_g,
        val_g,
        test_g,
        train_nfeat,
        train_labels,
        val_nfeat,
        val_labels,
        test_nfeat,
        test_labels,
    )

    run(args, device, data)
