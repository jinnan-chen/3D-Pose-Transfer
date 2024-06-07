import torch
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_scatter import scatter_add
# from scipy.sparse.csgraph import dijkstr
# from scipy.sparse import lil_matrix
# from scipy.spatial import Delaunay
# import torch_geometric as tg
from torch_geometric.utils import remove_self_loops
import torch.nn.functional as F
from torch.nn import Sequential, Dropout, Linear, ReLU, BatchNorm1d, Parameter



def MLP(channels, batch_norm=True):
    if batch_norm:
        return Sequential(*[Sequential(Linear(channels[i - 1], channels[i]), ReLU(), BatchNorm1d(channels[i], momentum=0.1))
                            for i in range(1, len(channels))])
    else:
        return Sequential(*[Sequential(Linear(channels[i - 1], channels[i]), ReLU()) for i in range(1, len(channels))])


class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, nn, aggr='max', **kwargs):
        super(EdgeConv, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn

    def forward(self, x, edge_index):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # import ipdb
        # ipdb.set_trace()
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return self.nn(torch.cat([x_i, (x_j - x_i)], dim=1))

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.out_channels)
        return aggr_out

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)



class GCUTPL(torch.nn.Module):
    def __init__(self, in_channels, out_channels, aggr='max'):
        super(GCUTPL, self).__init__()
        self.edge_conv_tpl = EdgeConv(in_channels=in_channels, out_channels=out_channels,
                                      nn=MLP([in_channels * 2, out_channels, out_channels]), aggr=aggr)
        # self.edge_conv_geo = EdgeConv(in_channels=in_channels, out_channels=out_channels // 2,
        #                           nn=MLP([in_channels * 2, out_channels // 2, out_channels // 2]), aggr=aggr)
        self.mlp = MLP([out_channels, out_channels])

    def forward(self, x, tpl_edge_index):
        x_tpl = self.edge_conv_tpl(x, tpl_edge_index)
        x_out = self.mlp(x_tpl)
        return x_out

class Skinpred(torch.nn.Module):
    def __init__(self, input_dim, num_part, aggr='max'):
        super(Skinpred, self).__init__()
        self.gcu_1 = GCUTPL(in_channels=input_dim, out_channels=64, aggr=aggr)
        self.gcu_2 = GCUTPL(in_channels=64, out_channels=128, aggr=aggr)
        self.gcu_3 = GCUTPL(in_channels=128, out_channels=256, aggr=aggr)
        self.mlp_glb = MLP([(64 + 128 + 256), 256])
        self.mlp2 = MLP([256, 256, 128])
        self.mlp3 = Linear(128, num_part)

    def forward(self, x, tpl_edge_index=None, batch=None, data=None, verbose=False):
        """
        treat the output as skinning weight instead of heatmap
        score: (N_all, K)
        weighted_pos: (bs, K, 3)
        """
        if data is not None:
            tpl_edge_index = data.tpl_edge_index
            batch = data.batch
        pos = x[:, :3]
        import ipdb
        ipdb.set_trace()
        x_1 = self.gcu_1(x, tpl_edge_index)
        x_2 = self.gcu_2(x_1, tpl_edge_index)
        x_3 = self.gcu_3(x_2, tpl_edge_index)
        x_4 = self.mlp_glb(torch.cat([x_1, x_2, x_3], dim=1))
        x = self.mlp2(x_4)
        x = self.mlp3(x)
        # softmax
        skinning_weights = torch.softmax(x, 1)
        score = skinning_weights / torch.repeat_interleave(scatter_sum(skinning_weights, batch, dim=0),
                                                           torch.bincount(batch), dim=0)

        weighted_pos = score[:, :, None] * pos[:, None]
        weighted_pos = scatter_sum(weighted_pos, batch, dim=0)

        return score, weighted_pos, x, skinning_weights