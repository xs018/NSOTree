import torch
import torch.nn as nn
import torch.nn.functional as F

import time 
import math
import numpy as np
from binarytree import build 
from dsplot.tree import BinaryTree
from utils.tree_utils import compute_logrank_test


def decimal_to_binary(decimal_num):
    binary_str = format(int(decimal_num), 'b')
    return binary_str

# activation functions
def my_softplus(x, tau=1., threshold=20.):
    truncate_mask = (x > threshold).type(torch.cuda.FloatTensor)
    return truncate_mask * x + (1. - truncate_mask) * (tau * torch.log(1 + torch.exp((1. - truncate_mask) * x / tau)))

def my_softplus_derivative(x, tau=1., threshold=20.):
    truncate_mask = (x > threshold).type(torch.cuda.FloatTensor)
    return truncate_mask + (1. - truncate_mask) / (1 + torch.exp(- (1. - truncate_mask) * x / tau) )

def my_gelu(x, approx='none'):
    if approx == 'none':
        x = 0.5 * x * (1 + torch.erf(x / math.sqrt(2)))
    elif approx == 'tanh':
        x = 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
    return x

def my_gelu_derivative(x, approx='none'):
    if approx == 'none':
        cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))
        pdf = torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
        deriv = 0.5 * (1.0 + cdf + x * pdf)
    elif approx == 'tanh':
        x_3 = (x + 0.044715 * torch.pow(x, 3))
        cdf = 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * x_3))
        pdf = torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
        deriv = 0.5 * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * x_3)) + \
                x * pdf * (1.0 - cdf**2) * (math.sqrt(2 / math.pi) + \
                0.5 * x * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * x_3)))
        
    return deriv


class MultiHeadAttention(nn.Module):
    """
    multi-head attention block
    """
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, gated=False):
        super(MultiHeadAttention, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(dim_V, num_heads)
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

        self.gate = None
        if gated:
            self.gate = nn.Sequential(nn.Linear(dim_Q, dim_V), nn.SiLU())

    def forward(self, Q, K):

        Q0 = Q

        Q = self.fc_q(Q).transpose(0, 1)
        K, V = self.fc_k(K).transpose(0, 1), self.fc_v(K).transpose(0, 1)

        A, _ = self.multihead_attn(Q, K, V)

        O = (Q + A).transpose(0, 1)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)

        if self.gate is not None:
            O = O.mul(self.gate(Q0))

        return O


class GAB(nn.Module):
    """
    equation (16) in the paper
    """

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(GAB, self).__init__()
        self.latent = nn.Parameter(torch.Tensor(1, num_inds, dim_out))  # low-rank matrix L

        nn.init.xavier_uniform_(self.latent)

        self.project_forward = MultiHeadAttention(dim_out, dim_in, dim_out, num_heads, ln=ln, gated=True)
        self.project_backward = MultiHeadAttention(dim_in, dim_out, dim_out, num_heads, ln=ln, gated=True)

    def forward(self, X):
        """
        This process, which utilizes 'latent_mat' as a proxy, has relatively low computational complexity.
        In some respects, it is equivalent to the self-attention function applied to 'X' with itself,
        denoted as self-attention(X, X), which has a complexity of O(n^2).
        """
        latent_mat = self.latent.repeat(X.size(0), 1, 1)
        H = self.project_forward(latent_mat, X)  # project the high-dimensional X into low-dimensional H
        X_hat = self.project_backward(X, H)  # recover to high-dimensional space X_hat

        return X_hat
    

class SoftThreshold(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass


class LowrankApprox(nn.Module):
    def __init__(self, in_features, out_features, bias=True, rank=1):
        super().__init__()
        self.rank = rank
        self.fc_v = nn.Linear(in_features, rank, False)
        self.vector_S = nn.Parameter(torch.empty((1, rank)))
        self.fc_U = nn.Linear(rank, out_features, bias)

    def forward(self, x):
        U, S, V = self.d_nsvd(x)
        x = self.fc_V(x)
        x = x.mul(self.vector_S)
        output = self.fc_U(x)
        return output

    def d_nsvd(self, x):
        U, S, V = torch.svd(x)
        S = S[:self.rank]
        U = U[:, :self.rank]  # * S.view(1, -1)
        V = V[:, :self.rank]  # * S.view(1, -1)
        V = torch.transpose(V, 0, 1)
        return U, S, V


class NeuralTree(nn.Module):
    def __init__(self, 
            input_dim, 
            output_dim, 
            hidden_dim, 
            num_layer, 
            num_back_layer, 
            dense = False, 
            drop_type = 'none', 
            net_type = 'locally_constant', \
            approx = 'none'):
        super(NeuralTree, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.num_back_layer = num_back_layer
        self.dense = dense
        self.drop_type = drop_type
        self.net_type = net_type
        self.n_neuron = self.hidden_dim * self.num_layer
        self.approx = approx
        self.num_nodes = 2 ** self.num_layer - 1
        
        self.layer = nn.ModuleList()

        self.weights = dict()
        self.biases = dict()
        self.output_constants = dict()
        self.output_weights = dict()
        self.output_biases = dict()

        accu_dim = input_dim
        for _ in range(self.num_layer):
            self.layer.append(nn.Linear(accu_dim, hidden_dim))

            if self.dense:
                accu_dim += hidden_dim
            else:
                accu_dim = hidden_dim

        if self.net_type == 'locally_constant':
            self.backward_layer = nn.ModuleList()
            backward_dim = 256
            cur_dim = self.num_layer * self.hidden_dim + self.num_layer * self.hidden_dim * self.input_dim
            for _ in range(self.num_back_layer):
                self.backward_layer.append(nn.Linear(cur_dim, backward_dim))
                cur_dim = backward_dim
            self.backward_layer.append(nn.Linear(cur_dim, output_dim))

        elif self.net_type == 'locally_linear':
            self.output_fc = nn.Linear(accu_dim, self.output_dim)
        else:
            print('net_type', self.net_type, 'is not supported')
            exit(0)

        # self.low_rank = GAB(dim_in=self.num_layer,
        #                 dim_out=int(self.num_layer * (1 - self.prune_ratio)),
        #                 num_heads=int(self.num_layer * (1 - self.prune_ratio)),
        #                 num_inds=1,
        #                 ln=True)

    def normal_forward(self, init_layer, p=0):
        assert(self.net_type == 'locally_linear')
        relu_masks = []
        if len(init_layer.shape) == 4:
            bz = init_layer.shape[0]
            init_layer = init_layer.view(bz, -1)
        cur_embed = init_layer
        batch_size = init_layer.shape[0]

        for i in range(self.num_layer):
            if self.training == False:
                next_embed = self.layer[i](cur_embed)
            else:
                w = self.layer[i].weight
                w = w.view(1, w.shape[0], w.shape[1]).expand(batch_size, -1, -1)

                b = self.layer[i].bias
                b = b.view(1, b.shape[0]).expand(batch_size, -1)

                next_embed = torch.bmm(F.dropout(w, p=p, training=self.training), cur_embed.unsqueeze(-1)) + b.unsqueeze(-1)
                next_embed = next_embed.squeeze(-1)

            relu_masks.append( (next_embed > 0) )
            next_embed = F.relu(next_embed)
            if self.dense:
                cur_embed = torch.cat((cur_embed, next_embed), 1)
            else:
                cur_embed = next_embed
        return self.output_fc(cur_embed), relu_masks

    def forward(self, init_layer, t=None, e=None, p=0, alpha=None, return_deriv=False):
        assert(self.net_type == 'locally_constant')
        if len(init_layer.shape) == 4:
            bz = init_layer.shape[0]
            init_layer = init_layer.view(bz, -1)

        logrank_test = (e is not None) and (t is not None)
        if logrank_test and isinstance(t, torch.Tensor) and isinstance(e, torch.Tensor):
            t, e = t.detach().cpu().numpy(), e.detach().cpu().numpy()
            # logrank_p_values = [None] * self.num_nodes

        cur_embed = init_layer # cur_embed is used for forward computation.
        x = init_layer.unsqueeze(-1) # x is used to compute Jacobian using dynamic programming.
        
        batch_size = x.shape[0]
        relu_masks = [np.ones((batch_size, ), dtype=bool)]
        patterns = []
        for i in range(self.num_layer):
            if self.drop_type != 'node_dropconnect':
                next_embed = self.layer[i](cur_embed)
               
            w = self.layer[i].weight
            w = w.view(1, w.shape[0], w.shape[1]).expand(batch_size, -1, -1)   
        
            if self.drop_type == 'node_dropconnect':
                b = self.layer[i].bias
                b = b.view(1, b.shape[0]).expand(batch_size, -1)

                next_embed = torch.bmm(F.dropout(w, p=p, training=self.training), cur_embed.unsqueeze(-1)) + b.unsqueeze(-1)
                next_embed = next_embed.squeeze(-1)
            else:
                pass

            split_mask = (next_embed > 0).squeeze(1).detach().cpu().numpy().astype(bool) # splitting mask 
            relu_masks.append(split_mask)
                    
            # <----------------- compute neuron derivatives ----------------->
            if self.approx == 'approx':
                neur_deriv = my_softplus_derivative(next_embed) 
                next_embed = my_softplus(next_embed) 
            elif self.approx == 'interpolation':
                neur_deriv = alpha * (next_embed > 0).type(torch.cuda.FloatTensor) + (1 - alpha) * my_softplus_derivative(next_embed)
                next_embed = alpha * F.relu(next_embed) + (1 - alpha) * my_softplus(next_embed)
            elif self.approx == 'none':
                neur_deriv = (next_embed > 0).type(torch.cuda.FloatTensor)
                next_embed = F.relu(next_embed)

            # print(next_embed.shape, neur_deriv.shape)
            patterns.append(neur_deriv)
            
            # <----------------- compute jacobians and offsets ----------------->
            neur_deriv = neur_deriv.unsqueeze(-1)
            if i == 0:
                jacobians = neur_deriv * w
                offsets = next_embed - torch.bmm(jacobians, x).squeeze(-1)
            else:
                if self.dense:
                    ite_jacobians = w[:, :, :self.input_dim] + torch.bmm(w[:, :, self.input_dim:], jacobians)
                else:
                    ite_jacobians = torch.bmm(w, jacobians[:, -self.hidden_dim:, :])

                ite_jacobians = neur_deriv * ite_jacobians
                ite_offsets = next_embed - torch.bmm(ite_jacobians, x).squeeze(-1)
                    
                # collect all derivatives
                jacobians = torch.cat([jacobians, ite_jacobians], dim=1)
                offsets = torch.cat([offsets, ite_offsets], dim=1)

            # update the next embed
            if self.dense:
                cur_embed = torch.cat((cur_embed, next_embed), 1)
            else:
                cur_embed = next_embed

        # print(jacobians.shape)

        if return_deriv:
            return jacobians, offsets

        if logrank_test:
            # <----------------- compute logrank test ----------------->
            logrank_p_values = compute_logrank_test(relu_masks, t, e)
            return logrank_p_values
        else:
            # <----------------- map leaf to output ----------------->
            leaf_input = torch.cat([jacobians, offsets.unsqueeze(-1)], dim=2).view(batch_size, -1) 

            # a feed-forward mapping 
            for i in range(self.num_back_layer):
                leaf_input = F.relu(self.backward_layer[i](leaf_input))
            leaf_output = self.backward_layer[-1](leaf_input)
            
            return leaf_output

    def build_tree(self, init_layer, t=None, e=None, p=0, alpha=None, topk=None, save_path=None):
        start = time.time()
        nodes = self._prepare_tree_nodes(topk)
        log_p_values = self.forward(init_layer, t, e, p, alpha)
        # reorganize tree
        nodes_reorganized = []
        assert len(nodes) == len(log_p_values)
        for n, p in zip(nodes, log_p_values):
            nodes_reorganized.append(n + f"\n [P: {p:.3f}]")

        if save_path:
            tree = BinaryTree(nodes=nodes_reorganized)
            tree.plot(output_path=save_path, shape='box')
        else:
            tree = build(nodes_reorganized)
        end = time.time()
        print(f"built tree in {end-start:.2}s")
        return nodes_reorganized

    def _prepare_tree_nodes(self, topk=None):
        if topk is None:
            topk = self.input_dim
        assert topk <= self.input_dim

        nodes = [None] * self.num_nodes

        for i in range(self.num_layer):
            node_mask = np.zeros((2 ** i, ))
            node_mask[1::2] = 1

            # <----------------- book-keeping oblique condition ----------------->
            b = self.layer[i].bias.data.detach().cpu().squeeze().numpy()
            w = self.layer[i].weight.data.detach().cpu().squeeze().numpy()
            
            # iterate all nodes at curret depth
            for node_n in range(2 ** i):
                cur_node_ind = (2 ** i - 1) + node_n # start_ind + node_n
                # we can take advatage of binary indicator
                binary_ind = list(decimal_to_binary(cur_node_ind+1)[1:])
                binary_ind = np.array([int(ind) for ind in binary_ind])

                if np.count_nonzero(binary_ind) == 0:
                    # very left node
                    w_x = w[:self.input_dim]
                    topk_ind = np.argsort(np.abs(w_x))[::-1][:topk]
                    w_plot = [f"{w_x[k]:.3f}*x{k}" for k in topk_ind]
                    w_plot = ' + '.join(w_plot) + f" + {b:.3f}"
                    w_plot = w_plot + " + ..." if self.input_dim > topk else w_plot
                    nodes[cur_node_ind] = w_plot
                else:
                    w_ind = np.concatenate([np.ones((self.input_dim, )), binary_ind])
                    topk_ind = np.argsort(np.abs(w))[::-1][:topk]
                    w_plot = []
                    for k in topk_ind:
                        if k < self.input_dim:
                            w_plot.append(f"{w[k]:.3f}*x{k}")
                        elif w_ind[k] == 1:
                            w_plot.append(f"{w[k]:.3f}*a{k-self.input_dim+1}")
                    w_plot = ' + '.join(w_plot) + f" + {b:.3f}"
                    w_plot = w_plot + " + ..." if self.input_dim > topk else w_plot
                    nodes[cur_node_ind] = w_plot

        return nodes
    
    