from torch import sign, tensor, no_grad
from torch.nn import (
    Module, Sequential, 
    Linear, Parameter
    )

# função sinal
class Sign(Module):
    def forward(self, x):
        return sign(x)

# modelo
model = Sequential()
model.add_module("layer_1", Linear(2, 2))
model.add_module("fun_1", Sign())
model.add_module("layer_2", Linear(2, 1))
model.add_module("fun_2", Sign())

# parâmetros
with no_grad():
    model.layer_1.weight = Parameter(tensor([[1., 1.], 
                                             [-1., -1.]]))
    model.layer_1.bias = Parameter(tensor([1., 1.]))
    model.layer_2.weight = Parameter(tensor([[1., 1.]]))
    model.layer_2.bias = Parameter(tensor([-1.]))

# dados de entrada
X = tensor([[1., 1.],
            [1., -1.],
            [-1., 1.],
            [-1., -1.]])

# inferência (propagação)
y_est = model(X)
print('Saída:')
print(y_est)
