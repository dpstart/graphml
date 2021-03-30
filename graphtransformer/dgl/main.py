import argparse
import torch
from train import train_iter, evaluate

from dgl.data import CoraGraphDataset
import dgl
from model import GraphTransformer

from scipy import sparse as sp


def add_lap_encodings(g, dim):

    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    eigval, eigvec = sp.linalg.eigs(L, k=dim + 1, which="SR", tol=1e-2)

    print(eigvec.shape)
    eigvec = eigvec[:, eigval.argsort()]
    g.ndata["lap_pos_enc"] = torch.from_numpy(eigvec[:, 1 : dim + 1]).float()
    return g


def run(args, g):

    model = GraphTransformer(args)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay
    )
    # scheduler = torch.optim.ReduceLROnPlateau(
    #     optimizer,
    #     mode="min",
    #     factor=args.lr_reduce_factor,
    #     patience=args.lr_schedule_patience,
    # )

    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_accs, epoch_val_accs = [], []

    for epoch in range(args.epochs):
        loss, acc, optimizer = train_iter(model, g, g.ndata["train_mask"], optimizer, args.device, epoch)

        if epoch % 10 == 0:

            eval_loss, eval_acc = evaluate(model, g, g.ndata["val_mask"], args.device)
            print(
                f"Epoch: {epoch} | Train Loss: {loss:.4f} | Train Acc: {acc:.4f} | Eval Loss: {eval_loss:.4f} | Eval Acc: {eval_acc:.4f}"
            )
    test_loss, test_acc = evaluate(model, g, g.ndata["test_mask"], args.device)
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Please give a config.json file with training/model/data/param details",
    )
    parser.add_argument("--model", help="Please give a value for model name")
    parser.add_argument("--out_dir", help="Please give a value for out_dir")
    parser.add_argument("--seed", help="Please give a value for seed")
    parser.add_argument("--epochs", help="Please give a value for epochs", default=100)
    parser.add_argument("--batch_size", help="Please give a value for batch_size")
    parser.add_argument(
        "--init_lr", help="Please give a value for init_lr", default=0.0007
    )
    parser.add_argument(
        "--lr_reduce_factor", help="Please give a value for lr_reduce_factor"
    )
    parser.add_argument(
        "--lr_schedule_patience", help="Please give a value for lr_schedule_patience"
    )
    parser.add_argument("--min_lr", help="Please give a value for min_lr")
    parser.add_argument(
        "--weight_decay", help="Please give a value for weight_decay", default=0.0
    )
    parser.add_argument(
        "--print_epoch_interval", help="Please give a value for print_epoch_interval"
    )
    parser.add_argument("--L", help="Please give a value for L")
    parser.add_argument(
        "--hidden_dim", help="Please give a value for hidden_dim", default=64
    )
    parser.add_argument("--out_dim", help="Please give a value for out_dim", default=64)
    parser.add_argument("--edge_feat", help="Please give a value for edge_feat")
    parser.add_argument("--readout", help="Please give a value for readout")
    parser.add_argument(
        "--num_heads", help="Please give a value for n_heads", default=8
    )
    parser.add_argument(
        "--in_feat_dropout", help="Please give a value for in_feat_dropout", default=0.1
    )
    parser.add_argument(
        "--dropout", help="Please give a value for dropout", default=0.1
    )
    parser.add_argument("--self_loop", help="Please give a value for self_loop")
    parser.add_argument("--max_time", help="Please give a value for max_time")
    parser.add_argument(
        "--pos_enc_dim", help="Please give a value for  pos_enc_dim", default=10
    )
    parser.add_argument(
        "--lap_pos_enc", help="Please give a value for lap_pos_enc", default=True
    )
    parser.add_argument(
        "--wl_pos_enc", help="Please give a value for wl_pos_enc", default=False
    )
    args = parser.parse_args()

    dataset = CoraGraphDataset()
    g = dataset[0]

    g = add_lap_encodings(g, args.pos_enc_dim)

    args.in_dim = 1433
    args.num_classes = 7

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    run(args, g)


if __name__ == "__main__":
    main()
