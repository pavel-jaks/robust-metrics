import torch
from metrics import StructuralDissimilarity

x = torch.Tensor([[[[1, 3], [1, 3]]]])
y = torch.Tensor([[[[3, 1], [3, 1]]]])
z = torch.Tensor([[[[20, 60], [20, 60]]]])

dssim = StructuralDissimilarity(l=1, window_size=2)

xy = dssim(x, y)
yz = dssim(y,z) 
xz = dssim(x, z)

print('Bingo' if xy + yz < xz else 'NotBingo')
print(f'xz:{xz.item()}; xy:{xy.item()}; yz:{yz.item()}')
