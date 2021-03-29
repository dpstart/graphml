import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphTransformerLayer
import dgl


class GraphTransformer(nn.Module):
    def __init__(self, args):

        super().__init__()

        in_dim_node = args["in_dim"]
        hidden_dim = args["hidden_dim"]
        out_dim = args["out_dim"]
        num_classes = args["num_classes"]
        num_heads = args["num_heads"]
        in_feat_dropout = args["in_feat_dropout"]
        dropout = args["dropout"]
        num_layers = args["num_layers"]

        self.readout = args["readout"]
        self.dropout = dropout
        self.num_classes = num_classes
        self.device = args["device"]
        self.lap_pos_enc = args["lap_pos_enc"]
        self.wl_pos_enc = args["wl_pos_enc"]

        max_wl_role_index = 100

        if self.lap_pos_enc:
            pos_enc_dim = args["pos_enc_dim"]
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embeding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)

        self.embedding_h = nn.Embedding(in_dim_node, hidden_dim)

        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layer = nn.ModuleList(
            [
                GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout)
                for _ in range(num_layers - 1)
            ]
        )

        self.layers.append(
            GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout)
        )

        self.mlp = MLPReadout(out_dim, num_classes)

    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None):

        h = self.embedding(h)

        if self.lap_pos_enc:

            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
            h = h + h_lap_pos_enc

        if self.wl_pos_enc:

            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc)
            h = h + h_wl_pos_enc

        h = self.in_feat_dropout(h)

        for layer in self.layers:

            h = layer(g, h)

        h_out = self.mlp(h)

        return h_out


    def loss(self, pred, label):

        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.num_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float()/V
        weight *= (cluster_sizes>0).float()


        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss
