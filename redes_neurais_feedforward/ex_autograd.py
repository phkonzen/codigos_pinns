from torch import tensor, sin, pi, ones_like

# entrada
x = tensor([0.,0.25,0.5,0.75,1.])
x = x.reshape(-1, 1)
x.requires_grad_(True)

# propagação
y = sin(pi * x)

# retropropagação (cálculo do gradiente)
y.backward(gradient=ones_like(y))

# print gradiente
print('dydx =', x.grad)
