import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.network_tree import NeuralTree

# # activation functions
# def my_softplus(x, tau=1., threshold=20.):
#     truncate_mask = (x > threshold).type(torch.cuda.FloatTensor)
#     return truncate_mask * x + (1. - truncate_mask) * (tau * torch.log(1 + torch.exp((1. - truncate_mask) * x / tau)))

# def my_softplus_derivative(x, tau=1., threshold=20.):
#     truncate_mask = (x > threshold).type(torch.cuda.FloatTensor)
#     return truncate_mask + (1. - truncate_mask) / (1 + torch.exp(- (1. - truncate_mask) * x / tau) )

# input = torch.linspace(-5, 5, 100).cuda()
# out = my_softplus(input).detach().cpu().numpy()
# derv = my_softplus_derivative(input).detach().cpu().numpy()
# print(out.shape, derv.shape)

# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# axes[0].plot(out)
# axes[1].plot(derv)
# plt.show()

# out = F.cosine_similarity(torch.randn(100, 4), torch.randn(3, 4))
# print(out.shape)

torch.manual_seed(0)

depth = 4
bsz = 128
model = NeuralTree(5, 1, 1, depth, 0, True).cuda()
x = torch.randn(bsz, 5).cuda()
t = torch.randn(bsz, )
e = torch.randint(0, 2, (bsz,) )
model(x, t, e)
model.traversal_tree()
model.fix_tree()

model(x, finetuing=True)