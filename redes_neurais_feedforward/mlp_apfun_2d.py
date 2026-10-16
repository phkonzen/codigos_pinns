import torch
import matplotlib.pyplot as plt

# dispositivo CPU/GPU?
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device = {device}')

# modelo
n_h = 2 # num de camadas escondidas
n_n = 30 # num de neurônios por camada escondida
act_fun = torch.nn.Tanh() # função de ativação

model = torch.nn.Sequential()
# camada de entrada
model.add_module('layer_1', torch.nn.Linear(2,n_n))
model.add_module('fun_1', act_fun)
# camadas escondidas
for i in range(n_h-1):
    model.add_module(f'layer_{i+2}', torch.nn.Linear(n_n,n_n))
    model.add_module(f'fun_{i+2}', act_fun)
# camada de saída
model.add_module(f'layer_{n_h+1}', torch.nn.Linear(n_n,1))
model.to(device) # envia modelo para o dispositivo

# treinamento

# fun obj
def fun(X):
    return torch.sin(torch.pi*X[:,0:1]) * \
        torch.sin(torch.pi*X[:,1:2])


# optimizador
optim = torch.optim.Adam(model.parameters(), lr=0.01)

# num de amostras por direção
ns_train = 400

# critério de parada
# num max épocas
nepochs = 50_000
# tolerância
tol = 1e-4

# amostras de validação
ns_x0_val = 20
ns_x1_val = 20
x0_v = torch.linspace(-1., 1., steps=ns_x0_val, device=device)
x1_v = torch.linspace(-1., 1., steps=ns_x1_val, device=device)
X0_v, X1_v = torch.meshgrid(x0_v, x1_v, indexing='ij')
X0_v = X0_v.reshape(-1,1)
X1_v = X1_v.reshape(-1,1)
X_val = torch.hstack((X0_v, X1_v))
y_vexp = fun(X_val) # valores esperados

for epoch in range(nepochs):

    model.train() # modo de treinamento

    # pontos de amostragem
    X = 2 * torch.rand(ns_train, 2, device=device) - 1.
    # valores esperados
    y_exp = fun(X) 
    
    # propagação (valores estimados)
    y_est = model(X)

    # função de perda
    loss = torch.mean((y_est - y_exp)**2)

    # retropropagação
    optim.zero_grad()
    loss.backward()
    optim.step()

    if epoch % 100 == 0:
        # validação
        model.eval() # modo de avaliação
        y_vest = model(X_val)
        loss_val = torch.mean((y_vest - y_vexp)**2)

        print(f'{epoch}: loss train: {loss.item():.4e}, val: {loss_val.item():.4e}')
    
        # critério de parada
        if (loss_val.item() < tol):
            print('Critério de parada atingido com sucesso!')
            break


# verificação
device = torch.device('cpu')
model.to(device)
model.eval() # modo de avaliação

# plot
fig, ax = plt.subplots()

x_plot = torch.linspace(-1., 1., steps=100)
y_plot = torch.linspace(-1., 1., steps=100)
X_plot, Y_plot = torch.meshgrid(x_plot, y_plot, indexing='ij')
X = torch.hstack((X_plot.reshape(-1,1), 
                    Y_plot.reshape(-1,1)))

 # valores esperados
y_exp = fun(X).reshape(100,100).numpy()
 # valores estimados
y_est = model(X).detach().reshape(100,100).numpy()

X_plot = X_plot.numpy()
Y_plot = Y_plot.numpy()

# função objetivo
levels = torch.linspace(y_exp.min(), y_exp.max(), 
                        steps=10).numpy()
cb = ax.contourf(X_plot, Y_plot, y_exp, 
                 levels=levels, cmap='viridis')
fig.colorbar(cb, ax=ax, label=r'$y$')

# MLP
cl = ax.contour(X_plot, Y_plot, y_est, 
                levels=levels, colors='k')

ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
plt.show()
