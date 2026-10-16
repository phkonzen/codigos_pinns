import torch

# modelo
model = torch.nn.Sequential()
model.add_module('layer_1', torch.nn.Linear(2,2))
model.add_module('fun_1', torch.nn.Tanh())
model.add_module('layer_2', torch.nn.Linear(2,1))
model.add_module('fun_2', torch.nn.Tanh())


# treinamento

# optimizador
optim = torch.optim.SGD(model.parameters(), lr=0.1)

# dados de treinamento
X_train = torch.tensor([[1., 1.],
                        [1., -1.],
                        [-1., 1.],
                        [-1., -1.]])
y_train = torch.tensor([-1., 1., 1., -1.]).reshape(-1,1)

print("\nDados de treinamento")
print("X_train =")
print(X_train)
print("y_train = ")
print(y_train)

# num max épocas
nepochs = 500

# tolerância
tol = 5e-2

for epoch in range(nepochs):

    # propagação
    y_est = model(X_train)

    # função erro
    loss = torch.mean((y_est - y_train)**2)

    print(f'{epoch}: {loss.item():.4e}')

    # critério de parada
    if (loss.item() < tol):
        break

    # retropropagação
    optim.zero_grad()
    loss.backward()
    optim.step()


# verificação
y = model(X_train)
print(f'y_est = {y}')
