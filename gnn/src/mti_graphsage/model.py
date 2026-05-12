from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MeanSAGELayer(nn.Module):
    """Full-batch GraphSAGE mean aggregation layer."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels * 2, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        src, dst = edge_index
        neigh_sum = x.new_zeros(x.size(0), x.size(1))
        neigh_sum.index_add_(0, dst, x[src])

        deg = x.new_zeros(x.size(0), 1)
        deg.index_add_(0, dst, torch.ones(dst.numel(), 1, dtype=x.dtype, device=x.device))
        neigh_mean = neigh_sum / deg.clamp_min(1.0)

        return self.linear(torch.cat([x, neigh_mean], dim=-1))


class GraphSAGEEncoder(nn.Module):
    """Two-layer-by-default GraphSAGE encoder for node embeddings."""

    def __init__(
        self,
        in_channels: int = 22,
        hidden_channels: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        channels = [in_channels] + [hidden_channels] * num_layers
        self.layers = nn.ModuleList(
            MeanSAGELayer(channels[i], channels[i + 1]) for i in range(num_layers)
        )
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = x
        for layer_idx, layer in enumerate(self.layers):
            h = layer(h, edge_index)
            if layer_idx != len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class EdgeDecoder(nn.Module):
    """MLP decoder over node-pair embeddings plus precomputed edge features."""

    def __init__(
        self,
        hidden_channels: int = 128,
        edge_attr_dim: int = 631,
        mlp_hidden_channels: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        decoder_in = hidden_channels * 4 + edge_attr_dim
        self.mlp = nn.Sequential(
            nn.Linear(decoder_in, mlp_hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_channels, mlp_hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_channels // 2, 1),
        )

    def forward(self, z: Tensor, edge_label_index: Tensor, edge_attr: Tensor) -> Tensor:
        src, dst = edge_label_index
        z_src = z[src]
        z_dst = z[dst]
        pair_features = torch.cat(
            [z_src, z_dst, z_src * z_dst, torch.abs(z_src - z_dst), edge_attr],
            dim=-1,
        )
        return self.mlp(pair_features).squeeze(-1)


class NodePairDecoder(nn.Module):
    """MLP decoder over source and destination node embeddings."""

    def __init__(
        self,
        hidden_channels: int = 128,
        mlp_hidden_channels: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        decoder_in = hidden_channels * 4
        self.mlp = nn.Sequential(
            nn.Linear(decoder_in, mlp_hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_channels, mlp_hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_channels // 2, 1),
        )

    def forward(self, z: Tensor, edge_label_index: Tensor) -> Tensor:
        src, dst = edge_label_index
        z_src = z[src]
        z_dst = z[dst]
        pair_features = torch.cat(
            [z_src, z_dst, z_src * z_dst, torch.abs(z_src - z_dst)],
            dim=-1,
        )
        return self.mlp(pair_features).squeeze(-1)


class GraphSAGENodePairPredictor(nn.Module):
    """GraphSAGE encoder plus node-pair decoder for dynamic negative sampling."""

    def __init__(
        self,
        node_feature_dim: int = 22,
        hidden_channels: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        decoder_hidden_channels: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = GraphSAGEEncoder(
            in_channels=node_feature_dim,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = NodePairDecoder(
            hidden_channels=hidden_channels,
            mlp_hidden_channels=decoder_hidden_channels,
            dropout=dropout,
        )

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.encoder(x, edge_index)

    def decode(self, z: Tensor, edge_label_index: Tensor) -> Tensor:
        return self.decoder(z, edge_label_index)

    def forward(self, x: Tensor, edge_index: Tensor, edge_label_index: Tensor) -> Tensor:
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


class GraphSAGELinkPredictor(nn.Module):
    """GraphSAGE encoder plus edge-feature-aware MLP decoder."""

    def __init__(
        self,
        node_feature_dim: int = 22,
        hidden_channels: int = 128,
        edge_attr_dim: int = 631,
        num_layers: int = 2,
        dropout: float = 0.3,
        decoder_hidden_channels: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = GraphSAGEEncoder(
            in_channels=node_feature_dim,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = EdgeDecoder(
            hidden_channels=hidden_channels,
            edge_attr_dim=edge_attr_dim,
            mlp_hidden_channels=decoder_hidden_channels,
            dropout=dropout,
        )

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.encoder(x, edge_index)

    def decode(self, z: Tensor, edge_label_index: Tensor, edge_attr: Tensor) -> Tensor:
        return self.decoder(z, edge_label_index, edge_attr)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_label_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index, edge_attr)
