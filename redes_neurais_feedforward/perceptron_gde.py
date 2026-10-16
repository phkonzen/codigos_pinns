import torch

# modelo

class Perceptron(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2,1)

    def forward(self, x):
        z =  self.linear(x)
        y = torch.tanh(z)
        return y

model = Perceptron()

# treinamento

# optimizador
optim = torch.optim.SGD(model.parameters(), lr=0.5)

# função erro
loss_fun = torch.nn.MSELoss()

# dados de treinamento
X_train = torch.tensor([[1., 1.],
                  [1., -1.],
                  [-1., 1.],
                  [-1., -1.]])
y_train = torch.tensor([1., -1., -1., -1.]).reshape(-1,1)

# num de amostras
ns = y_train.size(0)

print("\nDados de treinamento")
print("X_train =")
print(X_train)
print("y_train = ")
print(y_train)

# num max épocas
nepochs = 100
tol = 5e-2

for epoch in range(nepochs):

    model.train()
    for s in torch.randperm(ns):
        # propagação
        y_est_s = model(X_train[s:s+1,:])

        # função de perda da amostra
        loss_s = (y_est_s - y_train[s])**2

        # retropropagação
        optim.zero_grad()
        loss_s.backward()
        optim.step()

    # critério de parada
    model.eval()
    y_val = model(X_train)

    loss_val = loss_fun(y_val, y_train)
    print(f'{epoch}: {loss_val.item():.4e}')

    if (loss_val.item() < tol):
        break

# verificação
print(f'y = {y_val}')
