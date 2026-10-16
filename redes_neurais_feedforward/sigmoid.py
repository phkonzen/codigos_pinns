from torch import tensor
from torch.nn import Sigmoid
z = tensor([-5.0, -2.5, 0.0, 2.5, 5.0])
act_fun = Sigmoid()
act_fun(z)
