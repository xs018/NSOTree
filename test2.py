import torch
from math import floor


def round_to_nearest(n, ratio=1.0, base_number=4, allow_rank_eq1=False):
    rank = floor(floor(n * ratio) / base_number) * base_number
    rank = min(max(rank, 1), n)
    if rank == 1:
        rank = rank if allow_rank_eq1 else n
    return rank

# print(round_to_nearest(55))
x = torch.randn(100, 20)
U, S, V = torch.svd(x)
print(U.shape, S.shape, V.shape)


# for ep in range(epochs):
#     for data in dataloader:
#         # 1. compute g for whole dataset
#         g = 0
#         for data in dataloader:
#             # compute gradient for single batch

#         # 2. compute g_step

#         optimizer.step()