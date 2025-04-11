import torch
import mindspore as ms
import torch.nn as nn
import mindspore.nn as msnn
from mindspore import ops
import numpy as np
x = np.random.randn(3,5)
x1 = torch.tensor(x)
x2 = ms.tensor(x)
length1 = torch.tensor([[2,4,1],[3,1,0],[4,2,2]])
length2 = ms.tensor([[2,4,1],[3,1,0],[4,2,2]])
y1 = torch.gather(x1,dim=1,index = length1)
y2 = ops.gather_elements(x2,dim=1,index= length2)
print(y1)
print(y2)
