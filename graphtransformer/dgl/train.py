import torch
import torch.nn as nn
import math
import dgl


def accuracy(scores, targets):
    scores = scores.argmax(dim=1)
    acc = (scores==targets).float().sum().item()
    acc = acc / len(targets)
    return acc

def train_iter(model, g, mask,  optimizer, device, epoch):

    model.train()
    epoch_loss = epoch_train_acc = 0

    x = g.ndata["feat"].to(device)
    try:
        e = g.edata["feat"].to(device)
    except:
        e = None

    labels = g.ndata["label"].to(device)
    lap_pos_enc = g.ndata["lap_pos_enc"].to(device)
    optimizer.zero_grad()

    scores = model(
        g, x, e, lap_pos_enc, None
    )

    loss = model.loss(scores[mask], labels[mask])
    loss.backward(retain_graph=True)
    optimizer.step()
    epoch_loss += loss.item()
    epoch_train_acc += accuracy(scores[mask], labels[mask])
    return epoch_loss, epoch_train_acc, optimizer

def evaluate(model, g, mask, device):
    
    model.eval()
    x = g.ndata["feat"].to(device)
    try:
        e = g.edata["feat"].to(device)
    except:
        e = None

    labels = g.ndata["label"].to(device)
    lap_pos_enc = g.ndata["lap_pos_enc"].to(device)

    scores = model(
        g, x, e, lap_pos_enc, None
    )

    loss = model.loss(scores[mask], labels[mask])
    epoch_loss = loss.item()
    epoch_train_acc = accuracy(scores[mask], labels[mask])
    return epoch_loss, epoch_train_acc 

    
