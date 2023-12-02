import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import deepsnap
from deepsnap.hetero_gnn import forward_op
from torch_sparse import matmul
from sklearn.metrics import f1_score
from deepsnap.batch import Batch
from torch_sparse import SparseTensor
from typing import Tuple


class HeteroGNN(torch.nn.Module):
    def __init__(self, example_graph, params):
        super(HeteroGNN, self).__init__()
        self.hidden_size = params['hidden_size']

        self.convs1 = HeteroGNNWrapperConv(generate_convs(example_graph, HeteroGNNConv, self.hidden_size, first_layer=True), params)
        self.convs2 = HeteroGNNWrapperConv(generate_convs(example_graph, HeteroGNNConv, self.hidden_size, first_layer=False), params)

        self.bns1 = nn.ModuleDict({typ: nn.BatchNorm1d(self.hidden_size, eps=1) for typ in example_graph.node_types})
        self.bns2 = nn.ModuleDict({typ: nn.BatchNorm1d(self.hidden_size, eps=1) for typ in example_graph.node_types})
        self.relus1 = nn.ModuleDict({typ: nn.LeakyReLU() for typ in example_graph.node_types})
        self.relus2 = nn.ModuleDict({typ: nn.LeakyReLU() for typ in example_graph.node_types})
        self.post_mps = nn.ModuleDict({typ: nn.Linear(self.hidden_size, self.hidden_size) for typ in example_graph.node_types})

        ############# Your code here #############
        ## (~10 lines of code)
        ## Note:
        ## 1. For self.convs1 and self.convs2, call generate_convs at first and then
        ##    pass the returned dictionary of `HeteroGNNConv` to `HeteroGNNWrapperConv`.
        ## 2. For self.bns, self.relus and self.post_mps, the keys are node_types.
        ##    `deepsnap.hetero_graph.HeteroGraph.node_types` will be helpful.
        ## 3. Initialize all batchnorms to torch.nn.BatchNorm1d(hidden_size, eps=1).
        ## 4. Initialize all relus to nn.LeakyReLU().
        ## 5. For self.post_mps, each value in the ModuleDict is a linear layer
        ##    where the `out_features` is the number of classes for that node type.
        ##    `deepsnap.hetero_graph.HeteroGraph.num_node_labels(node_type)` will be
        ##    useful.
        ##########################################

    def forward(self, node_feature, edge_index):
        # TODO: Implement the forward function. Notice that `node_feature` is
        # a dictionary of tensors where keys are node types and values are
        # corresponding feature tensors. The `edge_index` is a dictionary of
        # tensors where keys are message types and values are corresponding
        # edge index tensors (with respect to each message type).

        x = node_feature
        idx = sparsify_edge_index(edge_index, node_feature=node_feature)

        ############# Your code here #############
        ## (~7 lines of code)
        ## Note:
        ## 1. `deepsnap.hetero_gnn.forward_op` can be helpful.

        x = self.convs1(node_features=x, edge_indices=idx)
        x = forward_op(x, self.bns1)
        x = forward_op(x, self.relus1)
        x = self.convs2(node_features=x, edge_indices=idx)
        x = forward_op(x, self.bns2)
        x = forward_op(x, self.relus2)
        ##########################################

        # Edge classification prediction head
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

    def loss(self, preds, y):
        loss = 0

        ############# Your code here #############
        ## (~3 lines of code)
        ## Note:
        ## 1. For each node type in preds, accumulate computed loss to `loss`
        ## 2. Loss need to be computed with respect to the given index
        ## 3. preds is a dictionary of model predictions keyed by node_type.
        ## 4. indeces is a dictionary of labeled supervision nodes keyed
        ##    by node_type

        for edge_type, y_vals in y.items():
            loss += F.cross_entropy(preds[edge_type], y_vals.to(torch.float))

        ##########################################

        return loss


class HeteroGNNConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels_src, in_channels_dst, out_channels):
        super(HeteroGNNConv, self).__init__(aggr="mean")

        self.in_channels_src = in_channels_src
        self.in_channels_dst = in_channels_dst
        self.out_channels = out_channels

        # To simplify implementation, please initialize both self.lin_dst
        # and self.lin_src out_features to out_channels
        self.lin_dst = nn.Linear(in_features=self.in_channels_dst, out_features=self.out_channels)
        self.lin_src = nn.Linear(in_features=self.in_channels_src, out_features=self.out_channels)

        self.lin_update = nn.Linear(in_features=self.out_channels * 2, out_features=self.out_channels)

        ############# Your code here #############
        ## (~3 lines of code)
        ## Note:
        ## 1. Initialize the 3 linear layers.
        ## 2. Think through the connection between the mathematical
        ##    definition of the update rule and torch linear layers!
        ##########################################

    def forward(
        self,
        node_feature_src,
        node_feature_dst,
        edge_index,
        size=None
    ):
        ############# Your code here #############
        ## (~1 line of code)
        ## Note:
        ## 1. Unlike Colabs 3 and 4, we just need to call self.propagate with
        ## proper/custom arguments.

        return self.propagate(edge_index=edge_index, size=size, node_feature_src=node_feature_src, node_feature_dst=node_feature_dst)

        ##########################################

    def message_and_aggregate(self, edge_index, node_feature_src):

        out = None
        ############# Your code here #############
        ## (~1 line of code)
        ## Note:
        ## 1. Different from what we implemented in Colabs 3 and 4, we use message_and_aggregate
        ##    to combine the previously seperate message and aggregate functions.
        ##    The benefit is that we can avoid materializing x_i and x_j
        ##    to make the implementation more efficient.
        ## 2. To implement efficiently, refer to PyG documentation for message_and_aggregate
        ##    and sparse-matrix multiplication:
        ##    https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html
        ## 3. Here edge_index is torch_sparse SparseTensor. Although interesting, you
        ##    do not need to deeply understand SparseTensor represenations!
        ## 4. Conceptually, think through how the message passing and aggregation
        ##    expressed mathematically can be expressed through matrix multiplication.

        out = matmul(edge_index, node_feature_src, reduce=self.aggr)


        ##########################################

        return out

    def update(self, aggr_out, node_feature_dst):

        ############# Your code here #############
        ## (~4 lines of code)
        ## Note:
        ## 1. The update function is called after message_and_aggregate
        ## 2. Think through the one-one connection between the mathematical update
        ##    rule and the 3 linear layers defined in the constructor.

       msgs = self.lin_src(aggr_out)
       owns = self.lin_dst(node_feature_dst)
       concats = torch.cat([owns, msgs], dim=1)
       aggr_out = self.lin_update(concats)

       ##########################################
       return aggr_out
    

class HeteroGNNWrapperConv(deepsnap.hetero_gnn.HeteroConv):
    def __init__(self, convs, params):
        super(HeteroGNNWrapperConv, self).__init__(convs, None)
        self.aggr = params["aggr"]

        # Map the index and message type
        self.mapping = {}

        # A numpy array that stores the final attention probability
        self.alpha = None

        self.attn_proj = None

        if self.aggr == "attn":
            ############# Your code here #############
            ## (~1 line of code)
            ## Note:
            ## 1. Initialize self.attn_proj, where self.attn_proj should include
            ##    two linear layers. Note, make sure you understand
            ##    which part of the equation self.attn_proj captures.
            ## 2. You should use nn.Sequential for self.attn_proj
            ## 3. nn.Linear and nn.Tanh are useful.
            ## 4. You can model a weight vector (rather than matrix) by using:
            ##    nn.Linear(some_size, 1, bias=False).
            ## 5. The first linear layer should have out_features as params['attn_size']
            ## 6. You can assume we only have one "head" for the attention.
            ## 7. We recommend you to implement the mean aggregation first. After
            ##    the mean aggregation works well in the training, then you can
            ##    implement this part.

            self.attn_proj = nn.Sequential(
                nn.Linear(in_features=params["hidden_size"], out_features=params["attn_size"], bias=True),
                nn.Tanh(),
                nn.Linear(in_features=params["attn_size"], out_features=1, bias=False),
            )

            ##########################################

    def reset_parameters(self):
        super(deepsnap.hetero_gnn.HeteroConv, self).reset_parameters()
        if self.aggr == "attn":
            for layer in self.attn_proj.children():
                layer.reset_parameters()

    def forward(self, node_features, edge_indices):
        message_type_emb = {}
        for message_key, message_type in edge_indices.items():
            src_type, edge_type, dst_type = message_key
            node_feature_src = node_features[src_type]
            node_feature_dst = node_features[dst_type]
            edge_index = edge_indices[message_key]
            message_type_emb[message_key] = (
                self.convs[message_key](
                    node_feature_src,
                    node_feature_dst,
                    edge_index,
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

    def aggregate(self, xs):
        # TODO: Implement this function that aggregates all message type results.
        # Here, xs is a list of tensors (embeddings) with respect to message
        # type aggregation results.

        if self.aggr == "mean":

            ############# Your code here #############
            ## (~2 lines of code)
            ## Note:
            ## 1. Explore the function parameter `xs`!

            x = torch.stack(xs, dim=-1)
            return x.mean(axis=-1)

            ##########################################

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
        

def generate_convs(hetero_graph, conv, hidden_size, first_layer=False):
    # TODO: Implement this function that returns a dictionary of `HeteroGNNConv`
    # layers where the keys are message types. `hetero_graph` is deepsnap `HeteroGraph`
    # object and the `conv` is the `HeteroGNNConv`.

    convs = {}

    ############# Your code here #############
    ## (~9 lines of code)
    ## Note:
    ## 1. See the hints above!
    ## 2. conv is of type `HeteroGNNConv`

    msgs = hetero_graph.message_types
    for m in msgs:
      if first_layer:
        in_ch_src = hetero_graph.num_node_features(m[0])
        in_ch_dst = hetero_graph.num_node_features(m[2])
      else:
        in_ch_src = hidden_size
        in_ch_dst = hidden_size
      convs.update({m: conv(in_channels_src=in_ch_src, in_channels_dst=in_ch_dst, out_channels=hidden_size)})

    ##########################################

    return convs


def train(model, optimizer, loader):
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        preds = model(batch.node_feature, batch.edge_index)
        loss = model.loss(preds=preds, y=batch.edge_label)
        loss.backward()
        optimizer.step()
    return loss.item()


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    num_ones = 0
    num_total = 0
    for batch in loader:
        preds = model(batch.node_feature, batch.edge_index)
        for edge_type in batch.edge_label:
            y_pred = preds[edge_type].round().cpu().detach().numpy()
            y_true = batch.edge_label[edge_type].cpu().detach().numpy()
            num_ones += y_pred.sum()
            num_total += len(y_pred)
            total += len(y_true)
            correct += sum(y_true == y_pred)
    return (correct/total, num_ones/num_total)


def sparsify_edge_index(edge_index: dict[torch.Tensor], node_feature: dict[torch.Tensor]) -> dict[Tuple: SparseTensor]:
    """Convert batch's edge index into sparsified version."""
    idx_dict = {}
    for key, cur_idx in edge_index.items():
        adj = SparseTensor(
            row=cur_idx[0],
            col=cur_idx[1],
            sparse_sizes=(node_feature[key[0]].shape[0], node_feature[key[2]].shape[0]),
        )
        idx_dict[key] = adj.t()
    return idx_dict
    