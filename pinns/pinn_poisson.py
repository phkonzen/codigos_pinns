import torch

# dispositivo GPU/CPU ?
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

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

## fonte
def f(X):
    return 2.0 * torch.pi**2 * torch.sin(torch.pi*X[:,0:1]) * \
           torch.sin(torch.pi*X[:,1:2])

## optimizador
optim = torch.optim.Adam(model.parameters(), lr=0.001)

## num de amostras por época
ns_x0 = 40
ns_x1 = 40

## num max épocas
nepochs = 50_000

## tolerância
tol = 1e-3
persist = 5

# contador de persistência
n_conv = 0

# laço de treinamento
for epoch in range(nepochs):

    model.train()

    # amostras de pontos internos
    x0 = 2.0 * torch.rand(ns_x0, 1, device=device) - 1.0
    x1 = 2.0 * torch.rand(ns_x1, 1, device=device) - 1.0
    X0, X1 = torch.meshgrid(x0.flatten(), x1.flatten(), indexing='ij')
    S = torch.hstack((X0.reshape(-1,1), X1.reshape(-1,1)))
    S.requires_grad = True
    
    # propagação nos pontos internos
    U = model(S)

    # derivadas de primeira ordem
    DU = torch.autograd.grad(U, S,
                            grad_outputs=torch.ones_like(U),
                            create_graph=True)[0]
    DUDX0 = DU[:,0:1] # du/dx0
    DUDX1 = DU[:,1:2] # du/dx1

    # derivadas de segunda ordem
    # d2u/dx0^2
    D2UDX0 = torch.autograd.grad(DUDX0, S,
                            grad_outputs=torch.ones_like(DUDX0),
                            create_graph=True)[0][:,0:1]
    # d2u/dx1^2
    D2UDX1 = torch.autograd.grad(DUDX1, S,
                            grad_outputs=torch.ones_like(DUDX1),
                            create_graph=True)[0][:,1:2]

    # laplaciano
    L = D2UDX0 + D2UDX1

    # fonte
    F = f(S)

    # função de perda: resíduo
    loss_res = torch.mean((F + L)**2)

    # amostras de pontos de contorno

    # contorno x1 = -1
    Scc1 = torch.hstack((x0, -torch.ones_like(x0)))
    Ucc1 = model(Scc1)
    loss_cc1 = torch.mean((Ucc1 - 0.0)**2)

    # contorno x0 = 1
    Scc2 = torch.hstack((torch.ones_like(x1), x1))
    Ucc2 = model(Scc2)
    loss_cc2 = torch.mean((Ucc2 - 0.0)**2)

    # contorno x1 = 1
    Scc3 = torch.hstack((x0, torch.ones_like(x0)))
    Ucc3 = model(Scc3)
    loss_cc3 = torch.mean((Ucc3 - 0.0)**2)

    # contorno x0 = -1
    Scc4 = torch.hstack((-torch.ones_like(x1), x1))
    Ucc4 = model(Scc4)
    loss_cc4 = torch.mean((Ucc4 - 0.0)**2)

    # função de perda total
    loss = loss_res + loss_cc1 + loss_cc2 + loss_cc3 + loss_cc4

    if (epoch % 100 == 0):
        print(f'{epoch}: loss_res: {loss.item():.2e}, cc1: {loss_cc1.item():.2e}, cc2: {loss_cc2.item():.2e}, cc3: {loss_cc3.item():.2e}, cc4: {loss_cc4.item():.2e}')
        torch.save(model, f'model.pt')

    if (loss.item() < tol):
        n_conv += 1
        print(f'\t{epoch}: loss: {loss.item():.2e}, {n_conv}/{persist}')
        if (n_conv >= persist):
            print(f'\t{epoch}: critério de parada atingido.')
            break
    else:
        n_conv = 0

    # backward
    optim.zero_grad()
    loss.backward()
    optim.step()  
