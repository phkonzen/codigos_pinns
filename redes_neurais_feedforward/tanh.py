from torch import tensor
from torch.nn import Tanh
z = tensor([-5.0, -2.5, 0.0, 2.5, 5.0])
act_fun = Tanh()
act_fun(z)
