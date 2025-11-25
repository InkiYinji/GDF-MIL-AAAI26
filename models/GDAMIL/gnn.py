'''
GNN models are implemented by torch_geometric
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_sparse import SparseTensor
from models.GDAMIL.conv_layers import GINConv, GATv2Conv
from models.GDAMIL.tad_tools import preprocess
from models.GAMIL import GAMIL


class DenseGNN(torch.nn.Module):
    def __init__(self, emb_dim, num_classes, gconv_type="gat_v2", initial_dim=256,
                 attention_head_num=1, number_of_layers=1, MLP_layernum=1,
                 average_nodes=2000, pooling_type="mean",
                 atten_hidden_size=256):
        super(DenseGNN, self).__init__()

        # FIXME: update number of layers
        self.preprocess = preprocess(
            emb_dim, initial_dim, attention_head_num, MLP_layernum, simple_distance='Y')

        if gconv_type == "gat_v2":
            self.convs_list = [
                GATv2Conv(
                    in_channels=attention_head_num*initial_dim,
                    out_channels=initial_dim,
                    edge_dim=2)
            ]

            for _ in range(number_of_layers - 1):
                self.convs_list.append(GATv2Conv(
                    in_channels=initial_dim,
                    out_channels=initial_dim,
                    edge_dim=2))
        else:
            raise NotImplementedError
        self.convs_list = nn.Sequential(*self.convs_list)
        self.number_of_layers = number_of_layers
        self.pooling_type = pooling_type

        self.graph_feature_dim = initial_dim * attention_head_num

        if self.pooling_type == "attention":
            self.attention_net = GAMIL(in_dim=initial_dim, num_classes=1)

        self.pred_layer = nn.Linear(self.graph_feature_dim, num_classes)
        # self.pred_layer = nn.Sequential(
        #     nn.Linear(self.graph_feature_dim, 32),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(32, num_classes)
        # )

    def gnn_forward(self, x, adj_t, edge_value, edge_index, edge_atten=None):
        # preprocess node feature and edge feature

        edge_index_sparse = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            value=edge_value,
            sparse_sizes=(x.shape[0], x.shape[0])
        )
        edge_attr = torch.zeros((edge_value.size(0), 2), device=edge_value.device)
        edge_attr[:, 0] = edge_value  # 第一列为权重/距离
        edge_attr[:, 1] = 0.0  # 第二列为角度（暂时设为0）

        x, adj_t, preprocess_edge_attr = self.preprocess(x, edge_index_sparse, edge_attr, None)

        # batch = graph_data.batch
        # row, col = adj_t.shape
        row, col, _ = adj_t.t().coo()
        # print(row)
        edge_index = torch.stack([row, col], dim=0)

        # graph_data.edge_index = edge_index

        # z = self.gnn_pool(x, graph_data.batch)

        x = self.convs_list[0](x=x, edge_index=edge_index, edge_attr=preprocess_edge_attr.to(x.device))
        x = F.relu(x)
        # # z = z + self.gnn_pool(x, graph_data.batch)
        z = self.gnn_pool(x, None)

        for i in range(self.number_of_layers - 1):
            x = F.dropout(x, p=0.2)
            x = self.convs_list[i+1](x=x, edge_index=edge_index, edge_attr=preprocess_edge_attr)
            x = F.relu(x)
            # z = z + self.gnn_pool(x, graph_data.batch)
            z = self.gnn_pool(x, None)

        return x, z

    def gnn_pool(self, x, batch):
        if self.pooling_type == "mean":
            x_pool = global_mean_pool(x, batch)
        elif self.pooling_type == "add":
            x_pool = global_add_pool(x, batch)
        elif self.pooling_type == "attention":
            A, h = self.attention_net(x)
            A = torch.transpose(A, 1, 0)      # KxN
            A = F.softmax(A, dim=1)           # softmax over N
            x_pool = A @ h
        else:
            raise NotImplementedError

        x_pool = F.dropout(x_pool, p=0.2)

        return x_pool

    def forward(self, x, adj, edge_value, edge_index, node_atten=None, edge_atten=None):

        if node_atten is not None:
            # add node attention initially
            x = x * node_atten

        x_pool, _ = self.gnn_forward(x, adj, edge_value, edge_index)

        logits = self.pred_layer(x_pool)
        probs = F.softmax(logits, dim=-1)

        return x_pool, logits, probs
