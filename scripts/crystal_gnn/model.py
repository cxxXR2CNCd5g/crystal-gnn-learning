import torch
import torch.nn as nn

from torch_geometric.nn import CGConv
from torch_geometric.nn import global_mean_pool

class CrystalGNN(nn.Module):
    def __init__(
            self,
            node_input_dim=100,
            edge_dim=21,
            hidden_dim=64
):
        super().__init__()

        self.input_layer = nn.Linear(
            node_input_dim,
            hidden_dim
        )

        self.conv1 = CGConv(
            channels=hidden_dim,
            dim=edge_dim,
            aggr="mean",
            batch_norm=True
        )

        self.conv2 = CGConv(
            channels=hidden_dim,
            dim=edge_dim,
            aggr="mean",
            batch_norm=True
        )

        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        x = torch.relu(self.input_layer(x))
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)

        graph_features = global_mean_pool(x, batch)

        predictions = self.output_layers(graph_features)

        return predictions.squeeze(-1)