import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch  # Assuming you are using PyTorch Geometric
from torch.nn import Linear, ReLU, Sequential, Dropout, BatchNorm1d
from torch_geometric.nn import GatedGraphConv, global_mean_pool
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import numpy as np



import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter



class MolGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_layers=6, hidden_dim=96, bb_dims=(180, 180, 180)):
        super(MolGNN, self).__init__()

        # Parameters
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Define GatedGraphConv for each graph component
        self.gated_conv1 = GatedGraphConv(out_channels=bb_dims[0], num_layers=num_layers)
        self.gated_conv2 = GatedGraphConv(out_channels=bb_dims[1], num_layers=num_layers)
        self.gated_conv3 = GatedGraphConv(out_channels=bb_dims[2], num_layers=num_layers)

        self.gated_conv12 = GatedGraphConv(out_channels=bb_dims[0]*2, num_layers=num_layers)
        self.gated_conv22 = GatedGraphConv(out_channels=bb_dims[1]*2, num_layers=num_layers)
        self.gated_conv32 = GatedGraphConv(out_channels=bb_dims[2]*2, num_layers=num_layers)

        self.gated_conv13 = GatedGraphConv(out_channels=bb_dims[0]*4, num_layers=num_layers)
        self.gated_conv23 = GatedGraphConv(out_channels=bb_dims[1]*4, num_layers=num_layers)
        self.gated_conv33 = GatedGraphConv(out_channels=bb_dims[2]*4, num_layers=num_layers)

        # Dropout and batch norm after pooling
        self.dropout = Dropout(0.1)
        self.graph_dropout = Dropout(0.1)

        fc_dim = (bb_dims[0] + bb_dims[1] + bb_dims[2])*4
        self.batch_norm = BatchNorm1d(fc_dim)

        # Fully connected layers
        self.fc1 = Linear(fc_dim, fc_dim * 3)
        self.fc2 = Linear(fc_dim * 3, fc_dim * 3)
        self.fc25 = Linear(fc_dim * 3, fc_dim)
        self.fc3 = Linear(fc_dim, 3)  # Output layer

    def forward(self, batch_data):
        # Debugging: Print the device of the input batch_data
        #print(f"batch_data device: {batch_data[0].x.device}")

        x1, edge_index1, batch1 = batch_data[0].x, batch_data[0].edge_index, batch_data[0].batch
        x2, edge_index2, batch2 = batch_data[1].x, batch_data[1].edge_index, batch_data[1].batch
        x3, edge_index3, batch3 = batch_data[2].x, batch_data[2].edge_index, batch_data[2].batch

        x1 = self.process_graph_component(x1, edge_index1, batch1, self.gated_conv1, self.gated_conv12, self.gated_conv13)
        x2 = self.process_graph_component(x2, edge_index2, batch2, self.gated_conv2, self.gated_conv22, self.gated_conv23)
        x3 = self.process_graph_component(x3, edge_index3, batch3, self.gated_conv3, self.gated_conv32, self.gated_conv33)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.batch_norm(x)

        # Apply dropout and fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc25(x)))
        x = self.fc3(x)

        return x

    def process_graph_component(self, x, edge_index, batch, conv_layer, conv_layer2, conv_layer3):
        x = F.relu(conv_layer(x, edge_index))
        x = self.graph_dropout(x)
        x = F.relu(conv_layer2(x, edge_index))
        x = self.graph_dropout(x)
        x = F.relu(conv_layer3(x, edge_index))
        x = global_mean_pool(x, batch)
        return x

class MolGNN2(torch.nn.Module):
    def __init__(self, num_node_features, num_layers=6, hidden_dim=96, bb_dims=(180, 180, 180)):
        super(MolGNN2, self).__init__()

        # Parameters
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Define GatedGraphConv for each graph component
        self.gated_conv1 = GatedGraphConv(out_channels=bb_dims[0], num_layers=num_layers)
        self.gated_conv2 = GatedGraphConv(out_channels=bb_dims[1], num_layers=num_layers)
        self.gated_conv3 = GatedGraphConv(out_channels=bb_dims[2], num_layers=num_layers)

        # Dropout and batch norm after pooling
        self.dropout = Dropout(0.1)
        self.graph_dropout = Dropout(0.1)

        fc_dim = bb_dims[0] + bb_dims[1] + bb_dims[2]
        self.batch_norm = BatchNorm1d(fc_dim)

        # Fully connected layers
        self.fc1 = Linear(fc_dim, fc_dim * 3)
        self.fc2 = Linear(fc_dim * 3, fc_dim * 3)
        self.fc25 = Linear(fc_dim * 3, fc_dim)
        self.fc3 = Linear(fc_dim, 3)  # Output layer

    def forward(self, batch_data):
        # Debugging: Print the device of the input batch_data
        #print(f"batch_data device: {batch_data[0].x.device}")

        x1, edge_index1, batch1 = batch_data[0].x, batch_data[0].edge_index, batch_data[0].batch
        x2, edge_index2, batch2 = batch_data[1].x, batch_data[1].edge_index, batch_data[1].batch
        x3, edge_index3, batch3 = batch_data[2].x, batch_data[2].edge_index, batch_data[2].batch

        x1 = self.process_graph_component(x1, edge_index1, batch1, self.gated_conv1)
        x2 = self.process_graph_component(x2, edge_index2, batch2, self.gated_conv2)
        x3 = self.process_graph_component(x3, edge_index3, batch3, self.gated_conv3)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.batch_norm(x)

        # Apply dropout and fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc25(x)))
        x = self.fc3(x)

        return x

    def process_graph_component(self, x, edge_index, batch, conv_layer):
        x = F.relu(conv_layer(x, edge_index))
        x = self.graph_dropout(x)
        x = global_mean_pool(x, batch)
        return x
    







    # i have removed all comments here to jepp it clean. refer to orginal link for code comments
# of MPNNModel
class MPNNLayer(MessagePassing):
	def __init__(self, emb_dim=64, edge_dim=4, aggr='add'):
		super().__init__(aggr=aggr)

		self.emb_dim = emb_dim
		self.edge_dim = edge_dim
		self.mlp_msg = nn.Sequential(
			nn.Linear(2 * emb_dim + edge_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
			nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU()
		)
		self.mlp_upd = nn.Sequential(
			nn.Linear(2 * emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
			nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU()
		)

	def forward(self, h, edge_index, edge_attr):
		out = self.propagate(edge_index, h=h, edge_attr=edge_attr)
		return out

	def message(self, h_i, h_j, edge_attr):
		msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
		return self.mlp_msg(msg)

	def aggregate(self, inputs, index):
		return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)

	def update(self, aggr_out, h):
		upd_out = torch.cat([h, aggr_out], dim=-1)
		return self.mlp_upd(upd_out)

	def __repr__(self) -> str:
		return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')


class MPNNModel(nn.Module):
	def __init__(self, num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1):
		super().__init__()

		self.lin_in = nn.Linear(in_dim, emb_dim)

		# Stack of MPNN layers
		self.convs = torch.nn.ModuleList()
		for layer in range(num_layers):
			self.convs.append(MPNNLayer(emb_dim, edge_dim, aggr='add'))

		self.pool = global_mean_pool

	def forward(self, data): #PyG.Data - batch of PyG graphs

		h = self.lin_in(data.x.float())  

		for conv in self.convs:
			h = h + conv(h, data.edge_index.long(), data.edge_attr.float())  # (n, d) -> (n, d)

		h_graph = self.pool(h, data.batch)  
		return h_graph

# our prediction model here !!!!
class Net(nn.Module):
	def __init__(self, ):
		super().__init__()

		self.output_type = ['infer'] #['infer', 'loss']

		graph_dim=48
		self.graph_encoder_1 = MPNNModel(
			 in_dim=8, edge_dim=4, emb_dim=graph_dim, num_layers=4,
		)

		self.graph_encoder_2 = MPNNModel(
			 in_dim=8, edge_dim=4, emb_dim=graph_dim, num_layers=4,
		)

		self.graph_encoder_3 = MPNNModel(
			 in_dim=8, edge_dim=4, emb_dim=graph_dim, num_layers=4,
		)

		fc_dim = graph_dim *3

		self.bind = nn.Sequential(
			nn.Linear(fc_dim, 1024),
			#nn.BatchNorm1d(1024),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(1024, 1024),
			#nn.BatchNorm1d(1024),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(1024, 512),
			#nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(512, 3),
		)

	def forward(self, batch):
		#graph = batch['graph']
		graph_batch_1, graph_batch_2, graph_batch_3 = batch[0], batch[1], batch[2]

		x1 = self.graph_encoder_1(graph_batch_1) 
		x2 = self.graph_encoder_2(graph_batch_2)
		x3 = self.graph_encoder_3(graph_batch_3)

		x = torch.cat([x1, x2, x3], dim=-1)

		output = self.bind(x)

		return output





### Graph Transformer with Gate Graph Convolution

class SwiGLU(nn.Module):
    def __init__(self, embed_dim, out_dim):
        super(SwiGLU, self).__init__()
        self.linear1 = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.linear2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear3 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.swish = nn.SiLU()

        # Linear layers used in adaptive_layer_norm
        self.linear_s1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_s2 = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, a, s):
        a = self.adaptive_layer_norm(a, s)
        b = self.linear1(a)
        b1, b2 = b.chunk(2, dim=-1)  # Split into two parts for element-wise multiplication
        b = self.swish(b1) * b2
        a = self.sigmoid(self.linear3(s)) * self.linear2(b)
        return a

    def adaptive_layer_norm(self, a, s):
        a = F.layer_norm(a, a.shape[-1:])
        s = F.layer_norm(s, s.shape[-1:])
        a = self.sigmoid(self.linear_s1(s)) * a + self.linear_s2(s)
        return a


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, out_dim, num_heads, dropout_prob=0.1, layer_norm = False):
        super(AttentionBlock, self).__init__()

        self.proj = nn.Linear(embed_dim, out_dim)
        self.attention = nn.MultiheadAttention(out_dim, num_heads, batch_first=True)
        self.swiglu = SwiGLU(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.norm1 = nn.LayerNorm(out_dim)
            self.norm2 = nn.LayerNorm(out_dim)
        

    def forward(self, x):
        # Self-attention layer
        
        x = self.proj(x)

        attn_output, _ = self.attention(x, x, x)
        attn_output = self.dropout(attn_output)
        if self.layer_norm:
            x = self.norm1(x + attn_output)
        else:
            x = x + attn_output
        
        # SwiGLU layer
        ff_output = self.swiglu(x, x)
        ff_output = self.dropout(ff_output)
        if self.layer_norm:
            x = self.norm2(x + ff_output)
        else:
            x = x + ff_output
        return x
    
class MLPBlock(nn.Module):
	def __init__(self, in_dim, out_dim, dropout_prob=0.1):
		super(MLPBlock, self).__init__()
		self.fc1 = nn.Linear(in_dim, out_dim)
		self.fc2 = nn.Linear(out_dim, out_dim)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(dropout_prob)

	def forward(self, x):
		x = self.dropout(self.relu(self.fc1(x)))
		x = self.dropout(self.relu(self.fc2(x)))
		return x
    
class MHALayer(MessagePassing):
    def __init__(self, node_emb, edge_emb, out_dims, num_heads, dropout=0.1, use_bias=True):
        super(MHALayer, self).__init__(aggr='add')

        self.node_emb = node_emb
        self.edge_emb = edge_emb
        
        self.out_dims = out_dims
        self.dropout = dropout

        self.attention = AttentionBlock(self.node_emb*2 + self.edge_emb, self.node_emb, num_heads, dropout, layer_norm=False)
        #self.attention_update = AttentionBlock(self.node_emb + self.edge_emb, self.out_dims, num_heads, dropout, layer_norm=False)
        self.mlp_update = MLPBlock(self.node_emb + self.edge_emb, self.out_dims, 0.05)

    
    def forward(self, h, edge_index, edge_attr):
        out = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, edge_attr):
        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self.attention(msg)

    def aggregate(self, inputs, index):
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)

    def update(self, aggr_out, h):
        if h.shape[0] > aggr_out.shape[0]:
            pad_size = h.shape[0] - aggr_out.shape[0]
            aggr_out = torch.cat([aggr_out, torch.zeros(pad_size, aggr_out.shape[1], device=aggr_out.device)], dim=0)
        elif h.shape[0] < aggr_out.shape[0]:
            pad_size = aggr_out.shape[0] - h.shape[0]
            h = torch.cat([h, torch.zeros(pad_size, h.shape[1], device=h.device)], dim=0)

        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_update(upd_out)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(emb_dim={self.node_emb}, aggr={self.aggr})'





class MHALayer(MessagePassing):
    def __init__(self, node_emb, edge_emb, out_dims, num_heads, dropout=0.1, use_bias=True):
        super(MHALayer, self).__init__(aggr='add')

        self.node_emb = node_emb
        self.edge_emb = edge_emb
        
        self.out_dims = out_dims
        self.dropout = dropout

        self.attention = AttentionBlock(self.node_emb*2 + self.edge_emb, self.node_emb, num_heads, dropout, layer_norm=False)
        self.attention_update = AttentionBlock(self.node_emb + self.edge_emb, self.out_dims, num_heads, dropout, layer_norm=False)

    
    def forward(self, h, edge_index, edge_attr):
        # print(f"h shape: {h.shape}, edge_index shape: {edge_index.shape}, edge_attr shape: {edge_attr.shape}")
        out = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, edge_attr):
        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self.attention(msg)
    
    def aggregate(self, inputs, index):
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)
    
    def update(self, aggr_out, h):
        if h.shape[0] > aggr_out.shape[0]:
            pad_size = h.shape[0] - aggr_out.shape[0]
            aggr_out = torch.cat([aggr_out, torch.zeros(pad_size, aggr_out.shape[1], device=aggr_out.device)], dim=0)
        elif h.shape[0] < aggr_out.shape[0]:
            pad_size = aggr_out.shape[0] - h.shape[0]
            h = torch.cat([h, torch.zeros(pad_size, h.shape[1], device=h.device)], dim=0)
    
        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.attention_update(upd_out)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(emb_dim={self.node_emb}, aggr={self.aggr})'


"""	def update(self, aggr_out, h):
		upd_out = torch.cat([h, aggr_out], dim=-1)
		#print(f"h shape: {h.shape},aggr_in shape: {aggr_out.shape}, aggred output shape: {upd_out.shape}")
		return self.attention_update(upd_out)
"""
class MHALayer(MessagePassing):
    def __init__(self, node_emb, edge_emb, out_dims, num_heads, dropout=0.1, use_bias=True):
        super(MHALayer, self).__init__(aggr='add')

        self.node_emb = node_emb
        self.edge_emb = edge_emb
        
        self.out_dims = out_dims
        self.dropout = dropout

        self.attention = AttentionBlock(self.node_emb*2 + self.edge_emb, self.node_emb, num_heads, dropout, layer_norm=False)
        self.attention_update = AttentionBlock(self.node_emb + self.edge_emb, self.out_dims, num_heads, dropout, layer_norm=False)

    
    def forward(self, h, edge_index, edge_attr):
        out = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, edge_attr):
        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self.attention(msg)

    def aggregate(self, inputs, index):
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)

    def update(self, aggr_out, h):
        if h.shape[0] > aggr_out.shape[0]:
            pad_size = h.shape[0] - aggr_out.shape[0]
            aggr_out = torch.cat([aggr_out, torch.zeros(pad_size, aggr_out.shape[1], device=aggr_out.device)], dim=0)
        elif h.shape[0] < aggr_out.shape[0]:
            pad_size = aggr_out.shape[0] - h.shape[0]
            h = torch.cat([h, torch.zeros(pad_size, h.shape[1], device=h.device)], dim=0)

        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.attention_update(upd_out)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(emb_dim={self.node_emb}, aggr={self.aggr})'




class GTBlock(nn.Module):
	def __init__(self, node_emb, edge_emb, out_dims, num_heads, num_layers, dropout_prob=0.1):
		super(GTBlock, self).__init__()
		self.node_embedding = nn.Linear(8, node_emb)
		self.edge_embedding = nn.Linear(4, edge_emb)  # Adjust if you're using edge features
		self.lap_pos_encoding_embedding = nn.Linear(4, node_emb)
		
		self.layers = nn.ModuleList([
			MHALayer(node_emb, edge_emb, out_dims, num_heads, dropout_prob) for _ in range(num_layers)
		])

	def forward(self, data):
		node_features, edge_index, edge_attr, lap_pos_enc = data.x, data.edge_index, data.edge_attr, data.eigens

		#Embeddings
		h = self.node_embedding(node_features)
		e = self.edge_embedding(edge_attr)
		lpe = self.lap_pos_encoding_embedding(lap_pos_enc)

		h = h + lpe

		# Convolution layers
		for layer in self.layers:
			h = layer(h, edge_index, e)

		#pooling
		#h = global_mean_pool(h, data.batch)

		return h

class GraphTransformer(nn.Module):
	def __init__(self, node_emb, edge_emb, out_dims, num_heads, num_layers, dropout_prob=0.1):
		super(GraphTransformer, self).__init__()

		self.bb1 = GTBlock(node_emb, edge_emb, out_dims, num_heads, 1, dropout_prob)
		self.bb2 = GTBlock(node_emb, edge_emb, out_dims, num_heads, 1, dropout_prob)
		self.bb3 = GTBlock(node_emb, edge_emb, out_dims, num_heads, 1, dropout_prob)

		self.gated1 = GatedGraphConv(out_dims, num_layers)
		self.gated2 = GatedGraphConv(out_dims, num_layers)
		self.gated3 = GatedGraphConv(out_dims, num_layers)

		fc_dim = node_emb * 3
		self.head = nn.Sequential(
			nn.Linear(fc_dim, fc_dim*4),
			#nn.BatchNorm1d(1024),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(fc_dim*4, fc_dim*4),
			#nn.BatchNorm1d(1024),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(fc_dim*4, fc_dim*2),
			#nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(fc_dim*2, 3),
		)

		#self.batch_norm = BatchNorm1d(node_emb)

	def forward(self, batch):
		#graph = batch['graph']
		graph_batch_1, graph_batch_2, graph_batch_3 = batch[0], batch[1], batch[2]

		x1 = self.bb1(graph_batch_1)
		x2 = self.bb2(graph_batch_2)
		x3 = self.bb3(graph_batch_3)

		x1 = self.process_gate(x1, graph_batch_1.edge_index, self.gated1, graph_batch_1.batch)
		x2 = self.process_gate(x2, graph_batch_2.edge_index, self.gated2, graph_batch_2.batch)
		x3 = self.process_gate(x3, graph_batch_3.edge_index, self.gated3, graph_batch_3.batch)

		x = torch.cat([x1, x2, x3], dim=-1)

		output = self.head(x)

		return output
	
	def process_gate(self, x, edge_index, conv, batch):
		x = F.relu(conv(x, edge_index))
		x = F.dropout(x, training=self.training)
		x = global_mean_pool(x, batch)
		return x







#### Simpler Graph Transformer

class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=4, aggr='add'):
        super().__init__(aggr=aggr)
        self.emb_dim = emb_dim
        self.edge_dim = emb_dim
        self.mlp_msg = nn.Sequential(
            nn.Linear(2 * emb_dim + edge_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU()
        )
        self.mlp_upd = nn.Sequential(
            nn.Linear(2 * emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
            nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU()
        )

    def forward(self, h, edge_index, edge_attr):
        out = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, edge_attr):
        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self.mlp_msg(msg)

    def aggregate(self, inputs, index):
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)

    def update(self, aggr_out, h):
        # Debugging print statements
        
        if h.shape[0] > aggr_out.shape[0]:
            pad_size = h.shape[0] - aggr_out.shape[0]
            aggr_out = torch.cat([aggr_out, torch.zeros(pad_size, aggr_out.shape[1], device=aggr_out.device)], dim=0)
        elif h.shape[0] < aggr_out.shape[0]:
            pad_size = aggr_out.shape[0] - h.shape[0]
            h = torch.cat([h, torch.zeros(pad_size, h.shape[1], device=h.device)], dim=0)

        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})'



class SwiGLU(nn.Module):
    def __init__(self, embed_dim, out_dim):
        super(SwiGLU, self).__init__()
        self.linear1 = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.linear2 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear3 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.swish = nn.SiLU()

        # Linear layers used in adaptive_layer_norm
        self.linear_s1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.linear_s2 = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, a, s):
        a = self.adaptive_layer_norm(a, s)
        b = self.linear1(a)
        b1, b2 = b.chunk(2, dim=-1)  # Split into two parts for element-wise multiplication
        b = self.swish(b1) * b2
        a = self.sigmoid(self.linear3(s)) * self.linear2(b)
        return a

    def adaptive_layer_norm(self, a, s):
        a = F.layer_norm(a, a.shape[-1:])
        s = F.layer_norm(s, s.shape[-1:])
        a = self.sigmoid(self.linear_s1(s)) * a + self.linear_s2(s)
        return a


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, out_dim, num_heads, dropout_prob=0.1, layer_norm = False):
        super(AttentionBlock, self).__init__()

        self.proj = nn.Linear(embed_dim, out_dim)
        self.attention = nn.MultiheadAttention(out_dim, num_heads, batch_first=True)
        self.swiglu = SwiGLU(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.norm1 = nn.LayerNorm(out_dim)
            self.norm2 = nn.LayerNorm(out_dim)
        

    def forward(self, x):
        # Self-attention layer
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.dropout(attn_output)
        if self.layer_norm:
            x = self.norm1(x + attn_output)
        else:
            x = x + attn_output
        
        # SwiGLU layer
        ff_output = self.swiglu(x, x)
        ff_output = self.dropout(ff_output)
        if self.layer_norm:
            x = self.norm2(x + ff_output)
        else:
            x = x + ff_output
        return x



class GTBlock(nn.Module):
	def __init__(self, node_emb, edge_emb, out_dims, num_heads, num_layers, dropout_prob=0.1):
		super(GTBlock, self).__init__()
		self.node_embedding = nn.Linear(8, node_emb)
		self.edge_embedding = nn.Linear(4, edge_emb)  # Adjust if you're using edge features
		self.lap_pos_encoding_embedding = nn.Linear(4, node_emb)

		self.node_attention = AttentionBlock(node_emb, node_emb, num_heads, dropout_prob, layer_norm=False)
		self.edge_attention = AttentionBlock(edge_emb, node_emb, num_heads, dropout_prob, layer_norm=False)
		
		self.layers = torch.nn.ModuleList()
		for layer in range(num_layers):
			self.layers.append(MPNNLayer(node_emb, edge_emb, aggr='add'))

	def forward(self, data):
		node_features, edge_index, edge_attr, lap_pos_enc = data.x, data.edge_index, data.edge_attr, data.eigens

		#Embeddings
		h = self.node_embedding(node_features)
		e = self.edge_embedding(edge_attr)
		lpe = self.lap_pos_encoding_embedding(lap_pos_enc)

		h = h + lpe

		h = self.node_attention(h)
		e = self.edge_attention(e)

		# Convolution layers
		for layer in self.layers:
			h = layer(h, edge_index, e)
               
		#pooling
		h = global_mean_pool(h, data.batch)

		return h

class GraphTransformerV2(nn.Module):
	def __init__(self, node_emb, edge_emb, out_dims, num_heads, num_layers, dropout_prob=0.1):
		super(GraphTransformerV2, self).__init__()

		self.bb1 = GTBlock(node_emb, edge_emb, out_dims, num_heads, num_layers, dropout_prob)
		self.bb2 = GTBlock(node_emb, edge_emb, out_dims, num_heads, num_layers, dropout_prob)
		self.bb3 = GTBlock(node_emb, edge_emb, out_dims, num_heads, num_layers, dropout_prob)

		fc_dim = node_emb * 3
		self.head = nn.Sequential(
			nn.Linear(fc_dim, fc_dim*4),
			#nn.BatchNorm1d(1024),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(fc_dim*4, fc_dim*4),
			#nn.BatchNorm1d(1024),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(fc_dim*4, fc_dim*2),
			#nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1),
			nn.Linear(fc_dim*2, 3),
		)

		#self.batch_norm = BatchNorm1d(node_emb)

	def forward(self, batch):
		#graph = batch['graph']
		graph_batch_1, graph_batch_2, graph_batch_3 = batch[0], batch[1], batch[2]

		x1 = self.bb1(graph_batch_1)
		x2 = self.bb2(graph_batch_2)
		x3 = self.bb3(graph_batch_3)
          
		x = torch.cat([x1, x2, x3], dim=-1)

		output = self.head(x)

		return output




import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool, MessagePassing
from torch_geometric.nn import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax
from torch import Tensor
from torch.nn import GRUCell, Parameter
from typing import Optional
from torch_geometric.typing import Adj, OptTensor
from torch_scatter import scatter

class GATEConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int, dropout: float = 0.0):
        super().__init__(aggr='add', node_dim=0)
        self.dropout = dropout
        self.att_l = Parameter(torch.empty(1, out_channels))
        self.att_r = Parameter(torch.empty(1, in_channels))
        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)
        self.bias = Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        alpha = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)
        out = self.propagate(edge_index, x=x, alpha=alpha)
        out = out + self.bias
        return out

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor, index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        alpha_j = (x_j @ self.att_l.t()).squeeze(-1)
        alpha_i = (x_i @ self.att_r.t()).squeeze(-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return self.lin2(x_j) * alpha.unsqueeze(-1)

class AttentiveFP(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, edge_dim: int, num_layers: int, num_timesteps: int, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout
        self.lin1 = Linear(in_channels, hidden_channels)
        self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim, dropout)
        self.gru = GRUCell(hidden_channels, hidden_channels)
        self.atom_convs = nn.ModuleList()
        self.atom_grus = nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout, add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))
        self.mol_conv = GATConv(hidden_channels, hidden_channels, dropout=dropout, add_self_loops=False, negative_slope=0.01)
        self.mol_conv.explain = False
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.gate_conv.reset_parameters()
        self.gru.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor) -> Tensor:
        x = F.leaky_relu_(self.lin1(x))
        h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()
        
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = conv(x, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu()
        
        
        out = global_add_pool(x, batch).relu_()
        new_edge_index = torch.arange(out.size(0), device=batch.device).unsqueeze(0).repeat(2, 1)

        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((out, out), new_edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.lin2(out)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(in_channels={self.in_channels}, hidden_channels={self.hidden_channels}, out_channels={self.out_channels}, edge_dim={self.edge_dim}, num_layers={self.num_layers}, num_timesteps={self.num_timesteps})'

class SelfAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)

        self.ff_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim*3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim*3, feature_dim)
        )

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        attention_scores = self.softmax(attention_scores)

        attended_values = torch.matmul(attention_scores, V)

        attended_values = self.ff_net(attended_values)
        return attended_values
    

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        graph_dim = 96
        num_layers = 4
        num_timesteps = 2 #6
        self.graph_encoder_1 = AttentiveFP(in_channels=8, hidden_channels=graph_dim, out_channels=graph_dim, edge_dim=4, num_layers=num_layers, num_timesteps=num_timesteps)
        self.graph_encoder_2 = AttentiveFP(in_channels=8, hidden_channels=graph_dim, out_channels=graph_dim, edge_dim=4, num_layers=num_layers, num_timesteps=num_timesteps)
        self.graph_encoder_3 = AttentiveFP(in_channels=8, hidden_channels=graph_dim, out_channels=graph_dim, edge_dim=4, num_layers=num_layers, num_timesteps=num_timesteps)
        self.attention = SelfAttention(graph_dim * 3)
        fc_dim = graph_dim * 3

        self.bind = nn.Sequential(
            nn.Linear(fc_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            #nn.Linear(1024, 1024),
            #nn.ReLU(inplace=True),
            #nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 3)
        )
        #self.batch_norm = nn.BatchNorm1d(fc_dim)

    def forward(self, batch):
        graph_batch_1, graph_batch_2, graph_batch_3 = batch[0], batch[1], batch[2]
        
        x1 = self.graph_encoder_1(graph_batch_1.x, graph_batch_1.edge_index, graph_batch_1.edge_attr, graph_batch_1.batch) 
        x2 = self.graph_encoder_2(graph_batch_2.x, graph_batch_2.edge_index, graph_batch_2.edge_attr, graph_batch_2.batch)
        x3 = self.graph_encoder_3(graph_batch_3.x, graph_batch_3.edge_index, graph_batch_3.edge_attr, graph_batch_3.batch)

        x = torch.cat([x1, x2, x3], dim=-1)
        #x = self.batch_norm(x)
        #a_x = self.attention(x)  # Apply self-attention across building blocks
        #a_x = a_x.view(a_x.size(0), -1)  
        output = self.bind(x)
        return output

