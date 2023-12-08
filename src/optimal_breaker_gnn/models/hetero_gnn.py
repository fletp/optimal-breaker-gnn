from typing import Tuple

import deepsnap
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from deepsnap.batch import Batch
from deepsnap.hetero_gnn import forward_op
from deepsnap.hetero_graph import HeteroGraph
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from torch_sparse import SparseTensor, matmul
from typing import Type


class HeteroGNN(torch.nn.Module):
    """Top-level definition of a heterogeneous graph neural network."""

    def __init__(self, example_graph: HeteroGraph, params: dict):
        """Create model parameters and layers.

        Args:
            example_graph: HeteroGraph with the same entity types and numbers
                of features for each entity as the graphs to be processed.
            params: dict of parameters for the structure of the model
        """
        super(HeteroGNN, self).__init__()

        self.example_graph = example_graph
        self.num_layers = params["num_layers"]
        self.hidden_size = params["hidden_size"]
        self.sparsify_index = params["sparsify_index"]
        self.edge_dim = params["edge_dim"]
        self.attn_heads = params["attn_heads"]
        self.add_self_loops = params["add_self_loops"]
        self.conv_layer = params["conv_layer"]

        self.convs, self.bns, self.relus = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        node_types = self.example_graph.node_types
        for i in range(self.num_layers):
            if i == 0:
                first_layer = True
            else:
                first_layer = False
            if self.conv_layer == "GATv2Conv":
                hid = self.attn_heads * self.hidden_size
            elif self.conv_layer == "HeteroGNNConv":
                hid = self.hidden_size
            else:
                raise NotImplementedError()
            cur_convs = self.generate_convs(conv_layer=self.conv_layer, first_layer=first_layer)
            self.convs.append(HeteroGNNWrapperConv(cur_convs, params))
            self.bns.append(nn.ModuleDict({typ: nn.BatchNorm1d(hid, eps=1) for typ in node_types}))
            self.relus.append(nn.ModuleDict({typ: nn.PReLU() for typ in node_types}))
        self.post_mps = nn.ModuleDict({typ: nn.Linear(hid, self.hidden_size) for typ in node_types})


    def generate_convs(
            self,
            conv_layer: str, 
            first_layer: bool=False
        ) -> dict[nn.Module]:
        """Generate graph convolutional layers for each message type.
        
        Args:
            conv_layer: type of a torch.nn.Module to be used for the graph
                convolutional layers (e.g. "torch.nn.Linear")
            first_layer: if True, this is the first layer, and the input dim
                should be adjusted accordingly.

        Returns:
            Dictionary of graph convolutional modules, one for each message type
            in the example graph.
        """
        convs = {}
        msgs = self.example_graph.message_types
        for m in msgs:
            if conv_layer == "GATv2Conv":
                if first_layer:
                    in_ch_src = self.example_graph.num_node_features(m[0])
                else:
                    in_ch_src = self.hidden_size * self.attn_heads
                cur_conv = pyg_nn.GATv2Conv(
                    in_channels=in_ch_src,
                    out_channels=self.hidden_size,
                    edge_dim=self.edge_dim,
                    heads=self.attn_heads,
                    add_self_loops=self.add_self_loops,
                )
            elif conv_layer == "HeteroGNNConv":
                if first_layer:
                    in_ch = self.example_graph.num_node_features(m[0]) + self.example_graph.num_edge_features(m)
                else:
                    in_ch = self.hidden_size + self.example_graph.num_edge_features(m)
                cur_conv = HeteroGNNConv(
                    in_channels=in_ch,
                    out_channels=self.hidden_size,
                    aggr="sum",
                )
            else:
                raise NotImplementedError()
            convs.update({m: cur_conv})
        return convs

    def forward(
            self,
            node_feature: dict[str: torch.Tensor],
            edge_index: dict[Tuple[str, str, str], torch.Tensor],
            edge_feature: dict[Tuple[str, str, str], torch.Tensor],
        ) -> torch.Tensor:
        """Evaluate the model in the forward direction.

        Args:
            node_feature: dict of node feature tensors keyed by node type name
            edge_index: dict of edge index tensors keyed by edge type name tuple
            edge_feature: dict of edge feature tensors keyed by edge type name
                tuple
        
        Returns: tensor of predictions for the edges of the graph.
        """
        x = node_feature
        if self.sparsify_index:
            idx = sparsify_edge_index(edge_index, node_feature=node_feature)
        else:
            idx = edge_index

        for i in range(self.num_layers):
            x = self.convs[i](node_features=x, edge_indices=idx, edge_features=edge_feature)
            x = forward_op(x, self.bns[i])
            x = forward_op(x, self.relus[i])

        x = forward_op(x, self.post_mps)
        edges = self.edge_head(node_features=x, edge_indices=edge_index)
        edges = {edge_type: F.sigmoid(edge_vals) for edge_type, edge_vals in edges.items()}
        return edges
    
    def edge_head(self, node_features: dict, edge_indices: dict) -> dict:
        """Inner-product head for edge-level prediction."""
        edge_dict = {}
        for edge_type, edge_index in edge_indices.items():
            node_type_u, node_type_v = edge_type[0], edge_type[2]
            embs_u = node_features[node_type_u][edge_index[0]]
            embs_v = node_features[node_type_v][edge_index[1]]
            batch_size = edge_index.shape[1]
            d = embs_u.shape[1]
            embs_u = embs_u.reshape(batch_size, 1, d)
            embs_v = embs_v.reshape(batch_size, d, 1)
            edge_dict[edge_type] = torch.matmul(embs_u, embs_v).squeeze((1, 2))
        return edge_dict

    def loss(
            self,
            preds: dict[Tuple[str, str, str]: torch.Tensor],
            y: dict[Tuple[str, str, str]: torch.Tensor],
        ) -> float:
        """Calculate loss between a set of predictions and labels.
        
        Args:
            preds: edge class predictions (float) by edge type
            y: edge class labels (int) by edge type

        Returns:
            float giving the loss
        """
        loss = 0
        for edge_type, y_vals in y.items():
            loss += F.cross_entropy(preds[edge_type], y_vals.to(torch.float))
        return loss


class HeteroGNNConv(pyg_nn.MessagePassing):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            aggr: str,
        ):
        super(HeteroGNNConv, self).__init__(aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin_update = nn.Linear(in_features=self.in_channels, out_features=self.out_channels)

    def forward(
        self,
        node_feature: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feature: torch.Tensor,
        size: int = None,
    ):
        return self.propagate(edge_index=edge_index, size=size, x=node_feature, edge_feature=edge_feature)


    def message(self, x_j, edge_feature):
        return torch.concat([x_j, edge_feature], dim=1)

    def update(self, aggr_out):
       aggr_out = self.lin_update(aggr_out)
       return aggr_out
    

class HeteroGNNWrapperConv(deepsnap.hetero_gnn.HeteroConv):
    """Wrapper which manages the aggregation of messages across multiple message
    types, each with their own graph convolutional layers."""

    def __init__(
            self,
            convs: dict[Tuple[str, str, str]: nn.Module],
            params: dict,
        ):
        super(HeteroGNNWrapperConv, self).__init__(convs, None)
        self.aggr = params["aggr"]

        # Map the index and message type
        self.mapping = {}

        # A numpy array that stores the final attention probability
        self.alpha = None
        self.attn_proj = None

        if self.aggr == "attn":
            self.attn_proj = nn.Sequential(
                nn.Linear(in_features=params["hidden_size"], out_features=params["attn_size"], bias=True),
                nn.Tanh(),
                nn.Linear(in_features=params["attn_size"], out_features=1, bias=False),
            )


    def reset_parameters(self):
        super(deepsnap.hetero_gnn.HeteroConv, self).reset_parameters()
        if self.aggr == "attn":
            for layer in self.attn_proj.children():
                layer.reset_parameters()

    def forward(
            self, 
            node_features: dict[str: torch.Tensor], 
            edge_indices: dict[Tuple[str, str, str]: torch.Tensor], 
            edge_features: dict[Tuple[str, str, str]: torch.Tensor],
        ) -> dict[str: torch.Tensor]:
        """Evaluate the message aggregation in the forward direction.
        
        Args:
            node_features: dict of node feature tensors keyed by node type name
            edge_indices: dict of edge index tensors keyed by edge type name tuple
            edge_features: dict of edge feature tensors keyed by edge type name
                tuple
        
        Returns: tensor of node embeddings to update the graph with
        """
        message_type_emb = {}
        for message_key in edge_indices:
            src_type, edge_type, dst_type = message_key
            node_feature = node_features[src_type]
            edge_index = edge_indices[message_key]
            edge_feature = edge_features[message_key]
            message_type_emb[message_key] = (
                self.convs[message_key](
                    node_feature,
                    edge_index,
                    edge_feature,
                )
            )
        node_emb = {dst: [] for _, _, dst in message_type_emb.keys()}
        mapping = {}
        for (src, edge_type, dst), item in message_type_emb.items():
            mapping[len(node_emb[dst])] = (src, edge_type, dst)
            node_emb[dst].append(item)
        self.mapping = mapping
        for node_type, embs in node_emb.items():
            if len(embs) == 1:
                node_emb[node_type] = embs[0]
            else:
                node_emb[node_type] = self.aggregate(embs)
        return node_emb

    def aggregate(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Aggregate all message type results."""
        if self.aggr == "mean":
            x = torch.stack(xs, dim=-1)
            return x.mean(axis=-1)

        elif self.aggr == "attn":
            N = xs[0].shape[0] # Number of nodes for that node type
            M = len(xs) # Number of message types for that node type

            x = torch.cat(xs, dim=0).view(M, N, -1) # M * N * D
            z = self.attn_proj(x).view(M, N) # M * N * 1
            z = z.mean(1) # M * 1
            alpha = torch.softmax(z, dim=0) # M * 1

            # Store the attention result to self.alpha as np array
            self.alpha = alpha.view(-1).data.cpu().numpy()

            alpha = alpha.view(M, 1, 1)
            x = x * alpha
            return x.sum(dim=0)


def train(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loader: DataLoader,
        device: torch.device,
    ) -> float:
    """Train the model for one epoch through the given dataloader."""
    model.train()
    for batch in loader:
        batch.to(device)
        preds = model(batch.node_feature, batch.edge_index, batch.edge_feature)
        loss = model.loss(preds=preds, y=batch.edge_label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()


def evaluate(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device
    ) -> Tuple[float, float]:
    """Evaluate current model on a dataloader of graphs."""
    model.eval()
    correct, total, num_ones, num_total = 0, 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            batch.to(device)
            preds = model(batch.node_feature, batch.edge_index, batch.edge_feature)
            for edge_type in batch.edge_label:
                y_pred = preds[edge_type].round().cpu().detach().numpy()
                y_true = batch.edge_label[edge_type].cpu().detach().numpy()
                num_ones += y_pred.sum()
                num_total += len(y_pred)
                total += len(y_true)
                correct += sum(y_true == y_pred)
        correct_frac = correct / total
        ones_frac = num_ones / num_total
    return (correct_frac, ones_frac)


def sparsify_edge_index(
        edge_index: dict[torch.Tensor],
        node_feature: dict[torch.Tensor]
    ) -> dict[Tuple: SparseTensor]:
    """Convert batch's edge index into a SparseTensor."""
    idx_dict = {}
    for key, cur_idx in edge_index.items():
        adj = SparseTensor(
            row=cur_idx[0],
            col=cur_idx[1],
            sparse_sizes=(node_feature[key[0]].shape[0], node_feature[key[2]].shape[0]),
        )
        idx_dict[key] = adj.t()
    return idx_dict
    