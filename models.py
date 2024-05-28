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
	
