import torch
import torch.nn as nn
import math
import dgl


def train_epoch(model, optimizer, device, dataloader, epoch):

    model.train()
    epoch_loss = epoch_train_acc = nb_data = gpu_mem = 0

    for iter, (batch_graphs, batch_labels) in enumerate(dataloader):

        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata["feat"].to(device)
        batch_e = batch_graphs.edata["feat"].to(device)

        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        try:
            batch_lap_pos_enc = batch_graphs.ndata["lap_pos_enc"].to(device)
            sign_flip = torch.rand(batch_lap_pos_enc.size(1)).to(device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            batch_lap_pos_enc = batch_lap_pos_enc * sign_flip.unsqueeze(0)
        except:
            batch_lap_pos_enc = None

        try:
            batch_wl_pos_enc = batch_graphs.ndata["wl_pos_enc"].to(device)
        except:
            batch_wl_pos_enc = None

        batch_scores = model(
            batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc
        )
