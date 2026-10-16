import torch
import matplotlib.pyplot as plt

# dispositivo CPU/GPU?
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device = {device}')

# modelo
model = torch.nn.Sequential()
model.add_module('layer_1', torch.nn.Linear(1,25))
model.add_module('fun_1', torch.nn.Tanh())
model.add_module('layer_2', torch.nn.Linear(25,1))
model.to(device)

# treinamento

# fun obj
fun = lambda x: torch.sin(torch.pi*x)
a = torch.tensor(-1., device=device)
b = torch.tensor(1., device=device)

# optimizador
optim = torch.optim.SGD(model.parameters(),
                        lr=0.1, momentum=0.9)

# num de amostras por época
ns = 100

# critério de parada
# num max épocas
nepochs = 50_000
# tolerância
tol = 1e-5
# persistência
ne_persist = 10
count_persist = 0

# amostras de validação
X_val = torch.linspace(a, b, steps=100, device=device)
X_val = X_val.reshape(-1,1) # pontos de validação
y_vexp = fun(X_val) # valores esperados

for epoch in range(nepochs):

    model.train() # modo de treinamento

    # amostras
    X = (b - a) * torch.rand((ns,1), device=device) + a
    y_exp = fun(X) # valores esperados
    
    # propagação
    y_est = model(X) # valores estimados

    # função de perda
    loss = torch.mean((y_est - y_exp)**2)

    # retropropagação
    optim.zero_grad()
    loss.backward()
    optim.step()

    # validação
    model.eval() # modo de avaliação
    y_vest = model(X_val)
    loss_val = torch.mean((y_vest - y_vexp)**2)

    if epoch % 100 == 0:
        print(f'{epoch}: loss train: {loss.item():.4e}, val: {loss_val.item():.4e}')
    
    # critério de parada
    if (loss_val.item() < tol):
        count_persist += 1
        print(f'{epoch}: loss train: {loss.item():.4e}, val: {loss_val.item():.4e}, persist: {count_persist}/{ne_persist}')
        if count_persist >= ne_persist:
            print('Critério de parada atingido com sucesso!')
            break
    else:
      count_persist = 0


# verificação
device = torch.device('cpu')
model.to(device)
model.eval() # modo de avaliação

# plot
fig, ax = plt.subplots()

x = torch.linspace(-1., 1., steps=100)
x = x.reshape(-1,1) # pontos de validação

y_exp = fun(x) # valores esperados
ax.plot(x, y_exp, label='fun')

y_est = model(x).detach() # valores estimados
ax.plot(x, y_est, ls='--', label='model')

ax.legend()
ax.grid()
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
