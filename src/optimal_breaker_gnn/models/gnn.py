import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear


class HGT_Model(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, metadata, num_layers, num_heads, dropout, gain, bias):

        super().__init__()
        
        self.num_layers = num_layers
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_dim)
            
        self.convs = torch.nn.ModuleList()
        self.bns =  torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_dim, hidden_dim, metadata, num_heads, group='sum')
            self.convs.append(conv)
            bns = torch.nn.BatchNorm1d(hidden_dim)
            self.bns.append(bns)
        
        self.linear = Linear(-1, output_dim)
        self.dropout = dropout

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=gain)
                m.bias.data.fill_(bias)

    def reset_parameters(self):
        for node_type in self.lin_dict:
            self.lin_dict[node_type].reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        
        x_dict = {
            node_type: F.relu(self.lin_dict[node_type](x))
            for node_type, x in x_dict.items()
        }  
        for i in range(self.num_layers):
            x_dict = self.convs[i](x_dict, edge_index_dict)
            x_dict = {
            node_type: F.dropout(F.relu(self.bns[i](x)), p=self.dropout, training=self.training)
            for node_type, x in x_dict.items()} 
            
        x = x_dict['breaker']
        out = self.linear(x)
            
        return out
    

def train(model, device, data_loader, optimizer, loss_fn):
    model.train()
    loss = 0

    for step, batch in enumerate(data_loader):
        batch = batch.to(device)
        if batch['busbar'].x[0].shape[0] == 1:
            pass
        else:
            optimizer.zero_grad()
            output = torch.squeeze(model(batch))
            y = batch['breaker'].y
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

    return loss.item()

def evaluate(model, device, data_loader, loss_fn):
    model.eval()
    
    correct = 0
    total = 0
    num_ones = 0
    num_total = 0
    for step, batch in enumerate(data_loader):
        batch = batch.to(device)
        if batch['busbar'].x[0].shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                y_pred = F.sigmoid(torch.squeeze(model(batch))).round().cpu().detach().numpy()
            y_true = batch['breaker'].y.cpu().detach().numpy()
            num_ones += y_pred.sum()
            num_total += len(y_pred)
            total += len(y_true)
            correct += sum(y_true == y_pred)

    return (correct/total, num_ones/num_total)


