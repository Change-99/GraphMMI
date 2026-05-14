from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class NodeInputEncoder(nn.Module):
    """Numeric node features + trainable ID/type/species embeddings."""

    def __init__(
        self,
        num_numeric_features: int,
        num_nodes: int,
        num_species: int = 4,
        id_embedding_dim: int = 32,
        type_embedding_dim: int = 8,
        species_embedding_dim: int = 8,
        out_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.id_embedding = nn.Embedding(num_nodes, id_embedding_dim) if id_embedding_dim > 0 else None
        self.type_embedding = nn.Embedding(2, type_embedding_dim) if type_embedding_dim > 0 else None
        self.species_embedding = nn.Embedding(num_species, species_embedding_dim) if species_embedding_dim > 0 else None
        input_dim = num_numeric_features + id_embedding_dim + type_embedding_dim + species_embedding_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: Tensor, node_type: Tensor, species_id: Tensor) -> Tensor:
        node_ids = torch.arange(x.size(0), device=x.device)
        species_id = species_id.clamp_min(0)
        parts = [x]
        if self.id_embedding is not None:
            parts.append(self.id_embedding(node_ids))
        if self.type_embedding is not None:
            parts.append(self.type_embedding(node_type))
        if self.species_embedding is not None:
            parts.append(self.species_embedding(species_id))
        return self.proj(torch.cat(parts, dim=-1))


class MeanSAGELayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels * 2, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        src, dst = edge_index
        neigh_sum = x.new_zeros(x.size(0), x.size(1))
        neigh_sum.index_add_(0, dst, x[src])
        degree = x.new_zeros(x.size(0), 1)
        degree.index_add_(0, dst, torch.ones(dst.numel(), 1, dtype=x.dtype, device=x.device))
        neigh_mean = neigh_sum / degree.clamp_min(1.0)
        return self.linear(torch.cat([x, neigh_mean], dim=-1))


class GraphSAGEEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        residual: bool = False,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([MeanSAGELayer(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor | None = None) -> Tensor:
        h = x
        for layer_idx, layer in enumerate(self.layers):
            h_in = h
            h = layer(h, edge_index)
            if self.layer_norm:
                h = self.norms[layer_idx](h)
            if self.residual:
                h = h_in + h
            if layer_idx != len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class GATv2Layer(nn.Module):
    """Full-batch GATv2 layer without torch_geometric dependency."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 2,
        concat: bool = True,
        dropout: float = 0.3,
        negative_slope: float = 0.2,
        use_edge_weight: bool = False,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.use_edge_weight = use_edge_weight
        self.lin_src = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.empty(heads, out_channels))
        self.bias = nn.Parameter(torch.zeros(heads * out_channels if concat else out_channels))
        nn.init.xavier_uniform_(self.att)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor | None = None) -> Tensor:
        loop = torch.arange(x.size(0), dtype=torch.long, device=x.device)
        src = torch.cat([edge_index[0], loop], dim=0)
        dst = torch.cat([edge_index[1], loop], dim=0)
        h_src = self.lin_src(x).view(-1, self.heads, self.out_channels)
        h_dst = self.lin_dst(x).view(-1, self.heads, self.out_channels)
        messages = h_src[src]
        att_input = F.leaky_relu(messages + h_dst[dst], negative_slope=self.negative_slope)
        scores = (att_input * self.att).sum(dim=-1)
        if self.use_edge_weight and edge_weight is not None:
            loop_w = torch.zeros(x.size(0), device=x.device)
            full_w = torch.cat([edge_weight, loop_w], dim=0)
            weight_bias = torch.log(full_w.clamp_min(1e-8))
            scores = scores + weight_bias.unsqueeze(-1).expand_as(scores)

        index = dst.unsqueeze(-1).expand_as(scores)
        max_per_dst = scores.new_full((x.size(0), self.heads), -torch.inf)
        max_per_dst.scatter_reduce_(0, index, scores, reduce="amax", include_self=True)
        exp_scores = torch.exp(scores - max_per_dst[dst])
        denom = scores.new_zeros(x.size(0), self.heads)
        denom.scatter_add_(0, index, exp_scores)
        alpha = exp_scores / denom[dst].clamp_min(1e-16)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = x.new_zeros(x.size(0), self.heads, self.out_channels)
        out.index_add_(0, dst, messages * alpha.unsqueeze(-1))
        if self.concat:
            return out.reshape(x.size(0), self.heads * self.out_channels) + self.bias
        return out.mean(dim=1) + self.bias


class GATv2Encoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.3,
        concat: bool = False,
        residual: bool = False,
        layer_norm: bool = False,
        use_edge_weight: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            layer_out = hidden_dim // heads if concat else hidden_dim
            self.layers.append(
                GATv2Layer(
                    in_channels=hidden_dim,
                    out_channels=layer_out,
                    heads=heads,
                    concat=concat,
                    dropout=dropout,
                    use_edge_weight=use_edge_weight,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor | None = None) -> Tensor:
        h = x
        for layer_idx, layer in enumerate(self.layers):
            h_in = h
            h = layer(h, edge_index, edge_weight=edge_weight)
            if self.layer_norm:
                h = self.norms[layer_idx](h)
            if self.residual:
                h = h_in + h
            if layer_idx != len(self.layers) - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class LinkDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        edge_attr_dim: int = 0,
        decoder_hidden_dim: int = 256,
        dropout: float = 0.3,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        input_dim = hidden_dim * 4 + edge_attr_dim
        self.edge_attr_dim = edge_attr_dim
        first_block: list[nn.Module] = [nn.Linear(input_dim, decoder_hidden_dim)]
        if layer_norm:
            first_block.append(nn.LayerNorm(decoder_hidden_dim))
        first_block.extend([nn.ReLU(), nn.Dropout(dropout)])
        second_block: list[nn.Module] = [nn.Linear(decoder_hidden_dim, decoder_hidden_dim // 2)]
        if layer_norm:
            second_block.append(nn.LayerNorm(decoder_hidden_dim // 2))
        second_block.extend([nn.ReLU(), nn.Dropout(dropout)])
        self.mlp = nn.Sequential(
            *first_block,
            *second_block,
            nn.Linear(decoder_hidden_dim // 2, 1),
        )

    def forward(self, z: Tensor, edge_label_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        src, dst = edge_label_index
        z_src = z[src]
        z_dst = z[dst]
        parts = [z_src, z_dst, z_src * z_dst, torch.abs(z_src - z_dst)]
        if self.edge_attr_dim:
            if edge_attr is None:
                raise ValueError("edge_attr is required because decoder was initialized with edge_attr_dim > 0.")
            parts.append(edge_attr)
        return self.mlp(torch.cat(parts, dim=-1)).squeeze(-1)


class GraphMMILinkPredictor(nn.Module):
    """Shared predictor; only the GNN encoder changes between GraphSAGE and GATv2."""

    def __init__(
        self,
        encoder_name: str,
        num_numeric_features: int,
        num_nodes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        edge_attr_dim: int = 0,
        dropout: float = 0.3,
        gat_heads: int = 2,
        gat_concat: bool = False,
        id_embedding_dim: int = 32,
        type_embedding_dim: int = 8,
        species_embedding_dim: int = 8,
        residual: bool = False,
        layer_norm: bool = False,
        decoder_layer_norm: bool = False,
        use_edge_weight: bool = False,
    ) -> None:
        super().__init__()
        self.input_encoder = NodeInputEncoder(
            num_numeric_features=num_numeric_features,
            num_nodes=num_nodes,
            id_embedding_dim=id_embedding_dim,
            type_embedding_dim=type_embedding_dim,
            species_embedding_dim=species_embedding_dim,
            out_dim=hidden_dim,
            dropout=dropout,
        )
        if encoder_name.lower() == "graphsage":
            self.gnn = GraphSAGEEncoder(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                residual=residual,
                layer_norm=layer_norm,
            )
        elif encoder_name.lower() == "gatv2":
            self.gnn = GATv2Encoder(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                heads=gat_heads,
                dropout=dropout,
                concat=gat_concat,
                residual=residual,
                layer_norm=layer_norm,
                use_edge_weight=use_edge_weight,
            )
        else:
            raise ValueError(f"Unknown encoder_name: {encoder_name}")
        self.decoder = LinkDecoder(
            hidden_dim=hidden_dim,
            edge_attr_dim=edge_attr_dim,
            dropout=dropout,
            layer_norm=decoder_layer_norm,
        )

    def encode(self, x: Tensor, node_type: Tensor, species_id: Tensor, edge_index: Tensor, edge_weight: Tensor | None = None) -> Tensor:
        h0 = self.input_encoder(x, node_type, species_id)
        return self.gnn(h0, edge_index, edge_weight=edge_weight)

    def decode(self, z: Tensor, edge_label_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        return self.decoder(z, edge_label_index, edge_attr)

    def forward(
        self,
        x: Tensor,
        node_type: Tensor,
        species_id: Tensor,
        edge_index: Tensor,
        edge_label_index: Tensor,
        edge_attr: Tensor | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        z = self.encode(x, node_type, species_id, edge_index, edge_weight=edge_weight)
        return self.decode(z, edge_label_index, edge_attr)
