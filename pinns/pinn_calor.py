import torch

# dispositivo GPU/CPU ?
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

# modelo

n_h = 4 # num de camadas escondidas
n_n = 50 # num de neurônios por camada escondida
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
def f(t, x):
    return (torch.pi**2 - 1) * torch.exp(-t) * torch.sin(torch.pi*x)

## optimizador
optim = torch.optim.Adam(model.parameters(), lr=0.001)

## num de amostras por época
ns_t = 20 # num de amostras no tempo
ns_x = 40 # num de amostras no espaço

## num max épocas
nepochs = 50_000

## tolerância
tol = 1e-4
persist = 5

# contador de persistência
n_conv = 0

# validação
t_v = torch.linspace(0, 1, 25, device=device)
x_v = torch.linspace(0, 1, 50, device=device)
T_v, X_v = torch.meshgrid(t_v, x_v, indexing='ij')
S_v = torch.hstack((T_v.reshape(-1,1), X_v.reshape(-1,1)))
S_v.requires_grad = True

# laço de treinamento
for epoch in range(nepochs):

    model.train()

    # amostras de pontos internos
    t = torch.rand(ns_t, 1, device=device)
    x = torch.rand(ns_x, 1, device=device)
    T, X = torch.meshgrid(t.flatten(), x.flatten(), indexing='ij')
    S = torch.hstack((T.reshape(-1,1), X.reshape(-1,1)))
    S.requires_grad = True
    
    # propagação nos pontos internos
    U = model(S)

    # derivadas de primeira ordem
    DU = torch.autograd.grad(U, S,
                            grad_outputs=torch.ones_like(U),
                            create_graph=True)[0]
    DUDT = DU[:,0:1] # du/dt
    DUDX = DU[:,1:2] # du/dx

    # derivadas de segunda ordem
    # d2u/dx^2
    D2UDX = torch.autograd.grad(DUDX, S,
                            grad_outputs=torch.ones_like(DUDX),
                            create_graph=True)[0][:,1:2]

    # rhs
    L = DUDT - D2UDX

    # fonte
    F = f(S[:,0:1], S[:,1:2])

    # função de perda: resíduo
    loss_res = torch.mean((F - L)**2)

    # condição inicial
    Sci = torch.hstack((torch.zeros_like(x), x))
    Uci = model(Sci)
    Uexp_ci = torch.sin(torch.pi*x)
    loss_ci = torch.mean((Uci - Uexp_ci)**2)

    # condições de contorno

    # contorno x = 0
    Scc1 = torch.hstack((t, torch.zeros_like(t)))
    Ucc1 = model(Scc1)
    loss_cc1 = torch.mean((Ucc1 - 0.0)**2)

    # contorno x = 1
    Scc2 = torch.hstack((t, torch.ones_like(t)))
    Ucc2 = model(Scc2)
    loss_cc2 = torch.mean((Ucc2 - 0.0)**2)

    # função de perda total
    loss = loss_res + loss_ci + loss_cc1 + loss_cc2    

    if (epoch % 100 == 0):
        print(f'{epoch}: loss_res: {loss.item():.2e}, ci: {loss_ci.item():.2e}, cc1: {loss_cc1.item():.2e}, cc2: {loss_cc2.item():.2e}')
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
