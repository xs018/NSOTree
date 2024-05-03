import torch
import torch.nn as nn
import torch.nn.functional as F

import time 
import math
import numpy as np
from binarytree import build 
from dsplot.tree import BinaryTree
from collections import deque
from lifelines.statistics import logrank_test


def print_p_value(p, thres):
    if p > thres:
        return f"p > {thres}"
    else:
        return f"p < {thres}"

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


class TreeNode:
    def __init__(self, current_idx=1, 
                 pvalue=0, mask=None, oblique_rule=None, 
                 left=None, right=None, parent=None):
        self.current_idx = current_idx
        self.pvalue = pvalue
        self.mask = mask
        self.oblique_rule = oblique_rule
        self.left = left
        self.right = right
        self.parent = parent

    def __repr__(self):
        return f"pvalue:{self.pvalue:.4f}"


class NeuralTree(nn.Module):
    def __init__(self, 
            input_dim, 
            output_dim, 
            hidden_dim, 
            num_layer, 
            num_back_layer, 
            backward_dim = 256,
            drop_type = 'none'):
        super(NeuralTree, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.num_back_layer = num_back_layer
        self.drop_type = drop_type
        self.n_neuron = self.hidden_dim * self.num_layer
        self.num_nodes = 2 ** self.num_layer - 1
        self.log_p_values = None
        
        self.layer = nn.ModuleList()

        self.weights = dict()
        self.biases = dict()
        self.output_constants = dict()
        self.output_weights = dict()
        self.output_biases = dict()
        
        # self.soft_thres = SoftThreshold()

        accu_dim = input_dim
        for _ in range(self.num_layer):
            self.layer.append(nn.Linear(accu_dim, hidden_dim))
            accu_dim += hidden_dim
            
        self.backward_layer = nn.ModuleList()
     
        cur_dim = self.num_layer * self.hidden_dim + self.num_layer * self.hidden_dim * self.input_dim
        for _ in range(self.num_back_layer):
            self.backward_layer.append(nn.Linear(cur_dim, backward_dim))
            cur_dim = backward_dim
        self.backward_layer.append(nn.Linear(cur_dim, output_dim))

    def forward(self, init_layer, t=None, e=None, p=0, finetuing=False):
        if len(init_layer.shape) == 4:
            bz = init_layer.shape[0]
            init_layer = init_layer.view(bz, -1)

        do_logrank_test = (e is not None) and (t is not None)
        if do_logrank_test and isinstance(t, torch.Tensor) and isinstance(e, torch.Tensor):
            t, e = t.detach().cpu().numpy(), e.detach().cpu().numpy()

        cur_embed = init_layer # cur_embed is used for forward computation.
        x = init_layer.unsqueeze(-1) # x is used to compute Jacobian using dynamic programming.
        
        batch_size = x.shape[0]
        patterns = []

        current_level = []
        counter = 0
        for i in range(self.num_layer):
            if self.drop_type != 'node_dropconnect':
                next_embed = self.layer[i](cur_embed)
               
            w = self.layer[i].weight
            w = w.view(1, w.shape[0], w.shape[1]).expand(batch_size, -1, -1)   
        
            if self.drop_type == 'node_dropconnect':
                b = self.layer[i].bias
                b = b.view(1, b.shape[0]).expand(batch_size, -1)

                next_embed = torch.bmm(F.dropout(w, p=p, training=(self.training and not finetuing)), cur_embed.unsqueeze(-1)) + b.unsqueeze(-1)
                next_embed = next_embed.squeeze(-1)
            else:
                pass
            
            split_mask = (next_embed > 0).squeeze(1).detach().cpu().numpy().astype(bool)

            if finetuing and (i < self.num_layer - 2):
                current_prune_nodes = self.pruned_nodes[i+2]
                if len(current_prune_nodes) > 0:
                    for node in current_prune_nodes:
                        next_embed[node.mask] = F.relu(next_embed[node.mask])
                    
            # <----------------- compute neuron derivatives ----------------->
            neur_deriv = my_softplus_derivative(next_embed) 
            next_embed = my_softplus(next_embed) 
            
            patterns.append(neur_deriv)

            w_np = self.layer[i].weight.squeeze().detach().cpu().numpy()
            b_np = self.layer[i].bias.squeeze().detach().cpu().numpy()
            oblique_rule = np.concatenate([w_np, np.array(b_np).reshape(-1, )], axis=0)

            # <----------------- compute jacobians and offsets ----------------->
            neur_deriv = neur_deriv.unsqueeze(-1)
            if i == 0:
                jacobians = neur_deriv * w
                offsets = next_embed - torch.bmm(jacobians, x).squeeze(-1)

                self.root = TreeNode(mask=np.ones((batch_size, ), dtype=bool))
                counter += 1

                self.root.current_idx = counter
                self.root.oblique_rule = oblique_rule
                current_level = [self.root]
            else:
                ite_jacobians = w[:, :, :self.input_dim] + torch.bmm(w[:, :, self.input_dim:], jacobians)
                ite_jacobians = neur_deriv * ite_jacobians
                ite_offsets = next_embed - torch.bmm(ite_jacobians, x).squeeze(-1)
                    
                # collect all derivatives
                jacobians = torch.cat([jacobians, ite_jacobians], dim=1)
                offsets = torch.cat([offsets, ite_offsets], dim=1)

                next_level = []
                for node in current_level:
                    node_mask = node.mask
                    # Add left child
                    node.left = TreeNode(mask=(1 - split_mask) * node_mask, parent=node)
                    counter += 1
                    node.left.current_idx = counter
                    next_level.append(node.left)
                    
                    # Add right child
                    node.right = TreeNode(mask=split_mask * node_mask, parent=node)
                    counter += 1
                    node.right.current_idx = counter
                    next_level.append(node.right)
                    
                    if do_logrank_test: 
                        t_left, e_left = t[node.left.mask], e[node.left.mask]
                        t_right, e_right = t[node.right.mask], e[node.right.mask]
                        results = logrank_test(t_left, t_right, event_observed_A=e_left, event_observed_B=e_right)
                        node.pvalue = results.p_value

                current_level = next_level

            # update the next embed
            cur_embed = torch.cat((cur_embed, next_embed), 1)

        # update the p_value for the final level
        if do_logrank_test: 
            for node in current_level:
                node_mask = node.mask
                
                mask_left = (1 - split_mask) * node_mask
                t_left, e_left = t[mask_left], e[mask_left]
                
                mask_right = split_mask * node_mask
                t_right, e_right = t[mask_right], e[mask_right]

                results = logrank_test(t_left, t_right, event_observed_A=e_left, event_observed_B=e_right)
                node.pvalue = results.p_value

        # <----------------- map leaf to output ----------------->
        leaf_input = torch.cat([jacobians, offsets.unsqueeze(-1)], dim=2).view(batch_size, -1) 

        # a feed-forward mapping 
        for i in range(self.num_back_layer):
            leaf_input = F.relu(self.backward_layer[i](leaf_input))
        leaf_output = self.backward_layer[-1](leaf_input)
            
        return leaf_output

    def traversal_tree(self, thres=0.05, prune_after_depth=2):
        queue = deque([self.root])
        pruned_nodes = {i+1: [] for i in range(self.num_layer)}
        current_node = 0
        while queue:
            current_node += 1
            current_depth = int(math.log2(current_node)) + 1
           
            node = queue.popleft()
            
            if (node.pvalue > thres or node == np.nan) and current_depth > prune_after_depth:
                pruned_nodes[current_depth].append(node)

            if node.left and node.right:
                queue.append(node.left)
                queue.append(node.right)

        self.pruned_nodes = pruned_nodes
        # print(self.pruned_nodes)
    
    def fix_tree(self):
        for param in self.layer.parameters():
            param.requires_grad = False