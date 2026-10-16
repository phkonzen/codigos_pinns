from torch import tensor, sin, pi, ones_like
from torch.autograd import grad

# entrada
x = tensor([0.,0.25,0.5,0.75,1.])
x = x.reshape(-1, 1)
x.requires_grad_(True)

# propagação
y = sin(pi * x)

# retropropagação y (gradiente de primeira ordem)
dydx = grad(y, x, grad_outputs=ones_like(y), 
            create_graph=True)[0]

# propagação dydx (gradiente de segunda ordem)
d2ydx2 = grad(dydx, x, grad_outputs=ones_like(dydx), 
              create_graph=True)[0]

# print gradiente
print('d2ydx2 =', d2ydx2)
