import torch

# modelo treinado
model = torch.load('model.pt',
                   weights_only=False).to('cpu')

# autograd MLP
X = torch.tensor([-1.,-0.5,0.,0.5,1.]).reshape(-1,1)
X.requires_grad = True

Y = model(X)

# gradiente de primeira ordem
DYDX = torch.autograd.grad(
    Y, X,
    grad_outputs=torch.ones_like(Y),
    create_graph=True)[0]

# gradiente de segunda ordem
D2YDX2 = torch.autograd.grad(
    DYDX, X,
    grad_outputs=torch.ones_like(DYDX))[0]
