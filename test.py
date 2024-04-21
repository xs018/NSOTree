from dsplot.tree import BinaryTree

# tree = BinaryTree(nodes=['-0.5028*X1 + -0.1886*X0', 1, None])
# tree.plot(shape='box')

import torch
from models.network import NeuralTree

torch.manual_seed(1)

depth = 3
input_dim = 5
batch_size = 6
model = NeuralTree(input_dim, 1, 1, depth, 0, True, 'none', approx='approx').cuda()
x = torch.randn(batch_size, input_dim).cuda()
t = torch.randn(batch_size, ).cuda()
e = torch.randint(0, 2, (batch_size, )).cuda()

# model.build_tree(x, t, e, topk=3, save_path="tree.png")
# print(model(x, t, e))
model(x)
