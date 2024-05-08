import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Sequential, Dropout, BatchNorm1d, Dropout
from torch_geometric.nn import GatedGraphConv, global_mean_pool, MessagePassing
import torch.nn.functional as F

class MolGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_layers = 6, hidden_dim = 96, do=0.05):
        super(MolGNN, self).__init__()

        # Parameters for GatedGraphConv layers
        self.num_layers = num_layers  # Number of gated graph conv layers #6
        self.hidden_dim = hidden_dim  # Hidden dimension size      #64
        self.output_dim = 1  # Output dimension size.  1

        # GatedGraphConv layers
        self.gated_conv = GatedGraphConv(out_channels=self.hidden_dim, num_layers=self.num_layers)

        self.dropout = Dropout(do)
        self.batch_norm = BatchNorm1d(self.hidden_dim)

        #self.dropout = nn.Dropout(dropout)

        self.fc1 = Linear(self.hidden_dim, self.hidden_dim*4)
        self.fc2 = Linear(self.hidden_dim*4, self.hidden_dim)
        self.fc3 = Linear(self.hidden_dim, self.output_dim)


    def forward(self, data):
        #data = data['mol_graph']
        x, edge_index = data.x, data.edge_index

        # Apply GatedGraphConv layers
        x = F.relu(self.gated_conv(x, edge_index))
        x = self.dropout(x)

        # Global mean pooling
        x = global_mean_pool(x, data.batch)  # Aggregate node features to graph-level
        x = self.batch_norm(x)

        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x


class MultiMolGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_layers = 6, hidden_dim = 96, dropout=0.25, output_dim=3):
        super(MultiMolGNN, self).__init__()

        # Parameters for GatedGraphConv layers
        self.num_layers = num_layers  # Number of gated graph conv layers #6
        self.hidden_dim = hidden_dim  # Hidden dimension size   #64
        self.output_dim = output_dim  # Output dimension size.  1

        # GatedGraphConv layers
        self.gated_conv = GatedGraphConv(out_channels=self.hidden_dim, num_layers=self.num_layers)

        self.dropout = Dropout(dropout)
        self.input_dropout = Dropout(0.125)
        self.batch_norm = BatchNorm1d(self.hidden_dim)

        #self.dropout = nn.Dropout(dropout)

        self.fc1 = Linear(self.hidden_dim, self.hidden_dim*4)
        self.fc2 = Linear(self.hidden_dim*4, self.hidden_dim)
        self.fc3 = Linear(self.hidden_dim, self.output_dim)


    def forward(self, data):
        #data = data['mol_graph']
        x, edge_index = data.x, data.edge_index
        #x = self.input_dropout(x)

        # Apply GatedGraphConv layers
        x = F.relu(self.gated_conv(x, edge_index))
        x = self.dropout(x)

        # Global mean pooling
        x = global_mean_pool(x, data.batch)  # Aggregate node features to graph-level
        x = self.batch_norm(x)

        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x





from torch_geometric.nn import GatedGraphConv, NNConv, global_mean_pool, GCNConv, GATConv, GATv2Conv
import torch
from torch.nn import Linear, ReLU, Sequential, Dropout, BatchNorm1d, Dropout
import torch.nn as nn
from torch_geometric.nn import GatedGraphConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import numpy as np

import torch.nn.init as init


class MultiHeadAttentionLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_heads, dropout=0.1, use_bias=True):
        super(MultiHeadAttentionLayer, self).__init__(aggr='add')  # Use 'add' for aggregating messages.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = dropout

        # Linear projection layers
        self.proj_q = nn.Linear(in_channels, out_channels * num_heads, bias=use_bias)
        self.proj_k = nn.Linear(in_channels, out_channels * num_heads, bias=use_bias)
        self.proj_v = nn.Linear(in_channels, out_channels * num_heads, bias=use_bias)

        self.att_drop = nn.Dropout(dropout)


    def forward(self, x, edge_index):
        # Split input features into Q, K, V
        Q = self.proj_q(x)
        K = self.proj_k(x)
        V = self.proj_v(x)

        # Now let's reshape
        Q = Q.view(-1, self.num_heads, self.out_channels)
        K = K.view(-1, self.num_heads, self.out_channels)
        V = V.view(-1, self.num_heads, self.out_channels)

        # Compute attention scores
        score = (Q @ K.transpose(-2, -1)) / np.sqrt(self.out_channels)
        score = F.softmax(score, dim=-1)
        score = self.att_drop(score)
        #print("score shape:", score.shape)

        # Propagate scores and aggregate
        x1 = score @ V
        x1 = x1.view(-1, self.num_heads * self.out_channels)
        out = self.propagate(edge_index, x=x1, size=None)
        out = out.view(-1, self.num_heads * self.out_channels)
        #print("out shape:", out.shape)

        return out


    def message(self, x_j, x_i):
        return x_i * x_j

class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, batch_norm=True, residual=True, use_bias=False):
        super(GraphTransformerLayer, self).__init__()

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads, dropout, use_bias)

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.batch_norm = batch_norm

        self.O = nn.Linear(out_dim, out_dim)

        if self.batch_norm:
            #self.batch_norm1 = nn.BatchNorm1d(out_dim)
            #self.batch_norm2 = nn.BatchNorm1d(out_dim)
            self.batch_norm1 = nn.LayerNorm(out_dim)
            self.batch_norm2 = nn.LayerNorm(out_dim)

        # FFN
        self.FFN_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_layer2 = nn.Linear(out_dim*2, out_dim)

        self.rnn = torch.nn.GRUCell(out_dim, out_dim)


    def forward(self, x, edge_index):
        x_in1 = x

        # Attention
        attn_out = self.attention(x, edge_index)
        x = attn_out.view(-1, self.out_channels)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.O(x)

        if self.residual:
            x = x_in1 + x
        else:
            x = attn_out

        if self.batch_norm:
            x = self.batch_norm1(x)

        x_in2 = x

        # FFN
        x = F.relu(self.FFN_layer1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.FFN_layer2(x)

        if self.residual:
            x = x_in2 + x
        else:
            x = x

        if self.batch_norm:
            x = self.batch_norm2(x)

        x = self.rnn(x, x)

        return x


class MolGraphTransformer(nn.Module):
    def __init__(self, n_layers, node_dim, edge_dim, hidden_dim, n_heads, in_feat_dropout, dropout, pos_enc_dim):
        super(MolGraphTransformer, self).__init__()

        self.linear_h = nn.Linear(node_dim, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, n_heads, dropout, True, True, False)
                                     for _ in range(n_layers - 1)])
        self.layers.append(GraphTransformerLayer(hidden_dim, hidden_dim, n_heads, dropout, True, True, False))

        self.batch_norm = BatchNorm1d(hidden_dim)

        self.hidden_dim = hidden_dim
        self.output_dim = 3  # Output dimension size

        self.dropout = nn.Dropout(dropout)

        self.fc1 = Linear(self.hidden_dim, self.hidden_dim*4)
        self.fc2 = Linear(self.hidden_dim*4, self.hidden_dim)
        self.fc3 = Linear(self.hidden_dim, self.output_dim)

        #self.apply(weights_init)


    def forward(self, data):
        node_features, edge_index = data.x, data.edge_index

        # Separate node features and lap_pos_enc
        #node_features, lap_pos_enc = x[:, :-1], x[:, -1].unsqueeze(-1)

        # Input embedding
        h = self.linear_h(node_features)
        h = self.in_feat_dropout(h)

        # Convolution layers
        for conv in self.layers:
            h = conv(h, edge_index)  # Adjust method signature and call based on GraphTransformerLayerPyG implementation

        #print("h shape:", h.shape)
        h = global_mean_pool(h, data.batch)
        x = self.batch_norm(h)

        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x
