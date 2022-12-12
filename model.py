import torch 
from torch import nn 
import torch.nn.functional as F 
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from layers import GVP, _norm_no_nan 


class FloodModel(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.seq_len = args.time_steps 
        self.in_dims = args.in_dims
        self.enc_dims = (args.s_h_dim, args.v_h_dim) 
        self.n_conv_layers = args.n_conv_layers
        self.processor = FloodGNN(self.in_dims, self.enc_dims, self.n_conv_layers)
        self.feat_pred = FeatPred(self.enc_dims, self.in_dims) 
        self.label_pred = LabelPred(self.enc_dims)
    
    def forward(self, graphs):
        out_labels = []
        out_s_feats = []
        out_v_feats = []
        label_losses = 0
        feat_losses = 0
        edge_index = graphs.edge_index 
        s_fixed, s, v = graphs.s_fixed, graphs.s, graphs.v
        s_h, v_h = s[:, 0], v[:, 0]
        wdfp = graphs.wdfp 
        wdfp_h = wdfp[:, 0]
        for i in range(1, self.seq_len + 1):
            s_o, v_o = self.processor(edge_index, s_fixed, s_h, v_h, wdfp_h)
            wdfp_h = self.label_pred(s_o, v_o)
            s_h, v_h = self.feat_pred(s_o, v_o) 

            out_labels.append(wdfp_h)
            out_s_feats.append(s_h)
            out_v_feats.append(v_h)

            label_losses += self.sum_over_node(wdfp_h, wdfp[:, i], graphs.s_batch).mean()
            feat_losses += (self.sum_over_node(s_h, s[:, i], graphs.s_batch, l1=True).mean() 
                            + self.sum_over_node(v_h, v[:, i], graphs.s_batch, l1=True).mean()) / 2
        
        loss = (label_losses + feat_losses) / self.seq_len
        out_labels = torch.stack(out_labels, dim=1)
        out_s_feats = torch.stack(out_s_feats, dim=1)
        out_v_feats = torch.stack(out_v_feats, dim=1)
        
        return out_labels, (out_s_feats, out_v_feats), loss

    def sum_over_node(self, targets, preds, index, l1=False):
        if l1: 
            return scatter(torch.e ** (targets -2) *torch.abs(targets - preds), index, dim=0, reduce='sum')
        return scatter(torch.abs(targets - preds), index, dim=0, reduce='sum')


class FloodGNN(nn.Module):
    def __init__(self, in_dims, out_dims, n_layers):
        super().__init__()
        self.in_dims = in_dims 
        self.out_dims = out_dims 
        self.n_layers = n_layers

        self.conv_layers = nn.ModuleList()

        for i in range(self.n_layers):
            self.conv_layers.append(FloodGNNLayer(self.out_dims if i else self.in_dims, 
                                                self.out_dims))
    
    def forward(self, g, s_fixed, s, v, wdfp):
        s = torch.cat([s_fixed, s, wdfp], dim=-1)

        for i, conv in enumerate(self.conv_layers):
            s, v = conv(g, s, v)
        
        return s, v 


class FloodGNNLayer(MessagePassing):
    
    def __init__(self, in_dims, out_dims, activations=(F.relu, torch.sigmoid)):
        super().__init__(node_dim=0) 
        self.in_dims = in_dims 
        self.out_dims = out_dims 
        self.aggr = 'add'

        self.n_encode = GVP(self.in_dims, self.out_dims)
        self.m_gvp = GVP([d *2 for d in self.out_dims], self.out_dims)
        self.u_gvp = GVP([d *2 for d in self.out_dims], self.out_dims, activations=activations)

    def forward(self, edge_index, s, v):
        n_nodes = s.shape[0]
        s, v = self.n_encode((s, v))
        s_out, v_out = self.propagate(edge_index, s=s, v=v, n_nodes=n_nodes)
        s_out = torch.cat([s, s_out], dim=1)
        v_out = torch.cat([v, v_out], dim=1)
        s_out, v_out = self.u_gvp((s_out, v_out))

        return s_out, v_out 
    
    def message(self, s_i, v_i, s_j, v_j):
        s_m_out = torch.cat([s_i, s_j], dim=1)
        v_m_out = torch.cat([v_i, v_j], dim=1)
        s_m_out, v_m_out = self.m_gvp((s_m_out, v_m_out))

        return s_m_out, v_m_out

    def aggregate(self, inputs, index, n_nodes):
        s_aggr = scatter(inputs[0], index, dim=0, dim_size=n_nodes,
                reduce=self.aggr)
        v_aggr = scatter(inputs[1], index, dim=0, dim_size=n_nodes,
                reduce=self.aggr)
        
        return s_aggr, v_aggr


class LabelPred(nn.Module):
    def __init__(self, in_dims):
        super().__init__()
        self.in_dims = in_dims 
        self.gvp_layer = GVP(self.in_dims, (self.in_dims[0] * 2, 0))
        self.ln = nn.Linear(self.in_dims[0]*2, 1)

    def forward(self, s, v):
        out = self.gvp_layer((s, v))
        pred = self.ln(out)

        return pred 


class FeatPred(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.in_dims = in_dims 
        self.out_dims = out_dims 
        self.gvp_layer = GVP(self.in_dims, self.out_dims, activations=(None, None))
    
    def forward(self, s_x, v_x):
        _, v_out = self.gvp_layer((s_x, v_x))
        s_out = _norm_no_nan(v_out)
        v_out = v_out / s_out.unsqueeze(-1)

        return s_out, v_out 







