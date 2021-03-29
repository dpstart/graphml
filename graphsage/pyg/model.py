import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv

from tqdm import tqdm


class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):

        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers-1):

            if i == 0:
                self.layers.append(SAGEConv(in_channels, hidden_channels))
            else:
                self.layers.append(SAGEConv(hidden_channels, hidden_channels))

        self.layers.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adjs):
        
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]] # target nodes are placed first
            x = self.layers[i]((x, x_target), edge_index)

            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all, subgraph_loader, device):

        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description("Evaluating")

        for i in range(self.num_layers):

            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.layers[i]((x, x_target), edge_index)

                if i!= self.num_layers-1:
                    x = F.relu(x)
                xs.append(x.cpu())

                pbar.update(batch_size)
            
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all



