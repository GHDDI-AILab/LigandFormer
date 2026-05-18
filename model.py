"""Ligandformer model implementation.

This file is a cleaned, standalone version of the self-attention HAG-Net code
used for the Ligandformer manuscript.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import BatchNorm1d, GELU, LeakyReLU, Linear, ModuleList, Sequential
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, OptPairTensor, Size
from torch_scatter import scatter, segment_csr

try:
    from torch_geometric.nn.conv.utils.helpers import expand_left
except ModuleNotFoundError:
    from torch_geometric.nn.aggr.base import expand_left


class SpatialAttentionConv(nn.Module):
    """Self-attention over atoms within each molecular graph."""

    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.attention_dim = 128

        self.query = Sequential(
            nn.GroupNorm(input_dim, input_dim, affine=True),
            nn.Conv1d(input_dim, self.attention_dim, kernel_size=1),
            LeakyReLU(0.1, inplace=True),
        )
        self.key = Sequential(
            nn.GroupNorm(input_dim, input_dim, affine=True),
            nn.Conv1d(input_dim, self.attention_dim, kernel_size=1),
            LeakyReLU(0.1, inplace=True),
        )
        self.val = Sequential(
            nn.GroupNorm(input_dim, input_dim, affine=True),
            nn.Conv1d(input_dim, output_dim, kernel_size=1),
            LeakyReLU(0.1, inplace=True),
        )
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=num_heads,
            kdim=self.attention_dim,
            vdim=output_dim,
            batch_first=True,
        )
        self.output_proj = Linear(self.attention_dim, output_dim)

    def forward(
        self,
        x: Tensor,
        batch_index: Tensor,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        query_v = self.query(x).squeeze(0).T
        key_v = self.key(x).squeeze(0).T
        val_v = self.val(x).squeeze(0).T

        graph_num = int(batch_index.max().item()) + 1 if batch_index.numel() else 0
        graph_counts = torch.bincount(batch_index, minlength=graph_num).tolist()

        reweighted = []
        attention_blocks = []
        start = 0
        for graph_count in graph_counts:
            end = start + graph_count
            graph_query = query_v[start:end].unsqueeze(0)
            graph_key = key_v[start:end].unsqueeze(0)
            graph_val = val_v[start:end].unsqueeze(0)
            graph_output, graph_attn = self.multihead_attn(
                graph_query,
                graph_key,
                graph_val,
                need_weights=return_attention,
                average_attn_weights=False,
            )
            reweighted.append(self.output_proj(graph_output.squeeze(0)))
            if return_attention:
                attention_blocks.append(graph_attn.mean(dim=1).squeeze(0))
            start = end

        reweighted_v = torch.cat(reweighted, dim=0)
        if not return_attention:
            return reweighted_v, None

        total_nodes = reweighted_v.shape[0]
        attention = reweighted_v.new_zeros((total_nodes, total_nodes))
        start = 0
        for block in attention_blocks:
            end = start + block.shape[0]
            attention[start:end, start:end] = block
            start = end
        return reweighted_v, attention


class SpatialAttentionBlockGIN(nn.Module):
    """Bottleneck projection plus atom self-attention for one Ligandformer block."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        viz_att: bool = False,
        num_heads: int = 1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.viz_att = viz_att

        self.bottle_neck = Sequential(
            nn.GroupNorm(input_dim, input_dim, affine=True),
            nn.Conv1d(input_dim, output_dim, kernel_size=1),
            LeakyReLU(0.1, inplace=True),
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=output_dim)
        self.leak_relu = LeakyReLU(0.1, inplace=True)
        self.spatial_attention = SpatialAttentionConv(
            input_dim=output_dim,
            output_dim=output_dim,
            num_heads=num_heads,
        )

    def forward(self, x: Tensor, batch_index: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = x.T.unsqueeze(0)
        x = self.bottle_neck(x)
        reweighted_x, attention = self.spatial_attention(
            x,
            batch_index,
            return_attention=self.viz_att,
        )
        x = reweighted_x.T.unsqueeze(0)
        x = self.leak_relu(x).squeeze(0).T
        x = self.layer_norm(x)
        if self.viz_att:
            return x, attention
        return x


class LearnedPositionalEncoding(nn.Module):
    MODE_POSITION = "MODE_POSITION"
    MODE_EXPAND = "MODE_EXPAND"
    MODE_ADD = "MODE_ADD"

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        max_len: int = 10000,
        mode: str = MODE_POSITION,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        if self.mode == self.MODE_EXPAND:
            self.weight = nn.Parameter(torch.Tensor(num_embeddings * 2 + 1, embedding_dim))
        else:
            self.weight = nn.Parameter(torch.Tensor(max_len, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == self.MODE_EXPAND:
            indices = torch.clamp(x, -self.num_embeddings, self.num_embeddings) + self.num_embeddings
            return F.embedding(indices.long(), self.weight)
        learnedpe = self.weight[: x.size(0), :]
        if self.mode == self.MODE_POSITION:
            return learnedpe
        if self.mode == self.MODE_ADD:
            return x + learnedpe
        raise NotImplementedError(f"Unknown mode: {self.mode}")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return self.pe[: x.size(0), :]


class HAGConv(MessagePassing):
    """GIN-like graph convolution with max and sum aggregation."""

    def __init__(
        self,
        nn_module: Callable,
        aggregation_methods: Optional[List[str]] = None,
        multiple_aggregation_merge_method: str = "sum",
        node_feature_update_method: str = "cat",
    ):
        super().__init__()
        self.nn = nn_module
        self.aggregation_methods = aggregation_methods or ["max", "sum"]
        self.multiple_aggregation_merge_method = multiple_aggregation_merge_method
        self.node_feature_update_method = node_feature_update_method
        self.reset_parameters()

    def reset_parameters(self) -> None:
        reset(self.nn)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)
        out = self.propagate(edge_index, x=x, size=size)
        x_r = x[1]
        if self.node_feature_update_method == "cat":
            out = torch.cat([x_r, out], dim=-1)
        else:
            raise ValueError(f"Unsupported update method: {self.node_feature_update_method}")
        return self.nn(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def aggregate(self, inputs: Tensor, index: Tensor, ptr=None, dim_size: Optional[int] = None) -> Tensor:
        if ptr is not None:
            ptr = expand_left(ptr, dim=self.node_dim, dims=inputs.dim())
            return segment_csr(inputs, ptr, reduce="sum")

        outputs = [
            scatter(inputs, index, dim=0, dim_size=dim_size, reduce=method)
            for method in self.aggregation_methods
        ]
        if self.multiple_aggregation_merge_method == "sum":
            merged = outputs[0]
            for output in outputs[1:]:
                merged = merged + output
            return merged
        raise ValueError(f"Unsupported aggregation merge method: {self.multiple_aggregation_merge_method}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nn={self.nn})"


class LigandFormer(nn.Module):
    """Ligandformer graph model for binary molecular property prediction."""

    def __init__(
        self,
        node_feature_dim: int,
        block_num: int = 3,
        embedding_dim: int = 75,
        conv_hidden_dim: int = 256,
        classifier_hidden_dim: int = 256,
        output_dim: int = 1,
        aggregation_methods: Optional[List[str]] = None,
        multiple_aggregation_merge_method: str = "sum",
        node_feature_update_method: str = "cat",
        readout_methods: str = "mean",
        pyramid_feature: bool = True,
        viz_att: bool = False,
        att_num_heads: int = 1,
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        if node_feature_dim <= 0:
            raise ValueError("node_feature_dim must be positive for Ligandformer inputs.")

        self.node_feature_dim = node_feature_dim
        self.embedding_dim = embedding_dim
        self.block_num = block_num
        self.aggregation_methods = aggregation_methods or ["max", "sum"]
        self.multiple_aggregation_merge = multiple_aggregation_merge_method
        self.node_feature_update_method = node_feature_update_method
        self.conv_input_dim = embedding_dim * 2 if node_feature_update_method == "cat" else embedding_dim
        self.conv_hidden_dim = conv_hidden_dim
        self.readout_methods = readout_methods
        self.pyramid_feature = pyramid_feature
        self.classifier_hidden_dim = classifier_hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.viz_att = viz_att
        self.att_num_heads = att_num_heads

        self.node_embedding = Sequential(
            Linear(node_feature_dim, embedding_dim),
            LeakyReLU(0.1, inplace=True),
            Linear(embedding_dim, embedding_dim),
        )

        self.conv_blocks = ModuleList()
        self.spatial_atts = ModuleList()
        for i in range(block_num):
            self.conv_blocks.append(
                HAGConv(
                    Sequential(
                        Linear(self.conv_input_dim, conv_hidden_dim),
                        LeakyReLU(),
                        BatchNorm1d(conv_hidden_dim, momentum=0.01),
                        Linear(conv_hidden_dim, embedding_dim),
                        LeakyReLU(),
                        BatchNorm1d(embedding_dim, momentum=0.01),
                    ),
                    aggregation_methods=self.aggregation_methods,
                    multiple_aggregation_merge_method=self.multiple_aggregation_merge,
                    node_feature_update_method=node_feature_update_method,
                )
            )
            self.spatial_atts.append(
                SpatialAttentionBlockGIN(
                    input_dim=(i + 2) * embedding_dim,
                    output_dim=embedding_dim,
                    viz_att=viz_att,
                    num_heads=att_num_heads,
                )
            )

        classifier_input_dim = (1 + block_num) * embedding_dim if pyramid_feature else embedding_dim
        self.dense_0 = Sequential(Linear(classifier_input_dim, classifier_hidden_dim), BatchNorm1d(classifier_hidden_dim, momentum=0.01), GELU())
        self.dense_1 = Sequential(Linear(classifier_hidden_dim, classifier_hidden_dim), BatchNorm1d(classifier_hidden_dim, momentum=0.01), GELU())
        self.dense_2 = Linear(classifier_hidden_dim, output_dim)

    def extract_graph_feature(self, data) -> Tuple[Tensor, Optional[List[Tensor]]]:
        x, edge_index, batch_index = data.x, data.edge_index, data.batch
        x = self.node_embedding(x.float())

        block_input = x
        block_fusion = x
        hiddens = [x]
        attention_maps = []

        for i in range(self.block_num):
            x = self.conv_blocks[i](x=block_input, edge_index=edge_index)
            block_input = x
            block_fusion = torch.cat((block_fusion, x), dim=1)
            if self.viz_att:
                reweighted, attention = self.spatial_atts[i](block_fusion, batch_index)
                attention_maps.append(attention)
            else:
                reweighted = self.spatial_atts[i](block_fusion, batch_index)
            x = block_input + reweighted
            hiddens.append(x)

        if self.pyramid_feature:
            pooled_features = []
            for hidden in hiddens:
                if self.training and self.dropout > 0:
                    hidden = F.dropout(hidden, p=self.dropout, training=True)
                pooled_features.append(scatter(hidden, batch_index, dim=0, reduce=self.readout_methods))
            graph_feature = torch.cat(pooled_features, dim=1)
        else:
            if self.training and self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=True)
            graph_feature = scatter(x, batch_index, dim=0, reduce=self.readout_methods)

        return graph_feature, attention_maps if self.viz_att else None

    def forward(self, data):
        graph_feature, attention_maps = self.extract_graph_feature(data)
        x = self.dense_0(graph_feature)
        if self.training and self.dropout > 0:
            x = F.dropout(x, p=self.dropout * 2, training=True)
        x = self.dense_1(x)
        if self.training and self.dropout > 0:
            x = F.dropout(x, p=self.dropout * 2, training=True)
        logits = self.dense_2(x)
        if self.viz_att:
            return logits, graph_feature, attention_maps
        return logits, graph_feature


HAG_NET_Self_Att = LigandFormer
