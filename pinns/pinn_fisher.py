import torch

# dispositivo GPU/CPU ?
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

# modelo

n_h = 4 # num de camadas escondidas
n_n = 50 # num de neurônios por camada escondida
act_fun = torch.nn.Tanh() # função de ativação

model = torch.nn.Sequential()
# camada de entrada/
model.add_module('layer_1', torch.nn.Linear(2,n_n))
model.add_module('fun_1', act_fun)
# camadas escondidas
for i in range(n_h-1):
    model.add_module(f'layer_{i+2}', torch.nn.Linear(n_n,n_n))
    model.add_module(f'fun_{i+2}', act_fun)
# camada de saída
model.add_module(f'layer_{n_h+1}', torch.nn.Linear(n_n,1))

# adiciona param lambda
model.lmbda = torch.nn.Parameter(12.0 * torch.rand(1, device=device))

model.to(device) # envia modelo para o dispositivo

# treinamento

## optimizador
optim = torch.optim.Adam(model.parameters(), lr=0.001)

## num de amostras por época
ns_t = 20 # num de amostras no tempo
ns_x = 40 # num de amostras no espaço

## num max épocas
nepochs = 50_000

## critério de parada (lambda)
window_tol = 1e-3
window_size = 100
lmbda_window = [model.lmbda.item()]

# dados
ts_data = torch.tensor([0.1, 0.2, 0.3], device=device)
xs_data = torch.tensor([0.25, 0.5, 0.75], device=device)
Ts_data, Xs_data = torch.meshgrid(ts_data, xs_data, indexing='ij')
S_data = torch.hstack((Ts_data.reshape(-1,1), Xs_data.reshape(-1,1)))
lmbda_exp = torch.tensor(6.0, device=device)
Us_data = 1.0/(1.0 + torch.exp(torch.sqrt(lmbda_exp/6.0)*S_data[:,1:2] \
            - 5.0/6.0*lmbda_exp*S_data[:,0:1]))**2


# laço de treinamento
for epoch in range(nepochs):

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

    # param
    lmbda = model.lmbda

    # resíduo
    R = DUDT - D2UDX - lmbda*U*(1.0 - U)

    # função de perda: resíduo
    loss_res = torch.mean(R**2)

    # condição inicial
    Sci = torch.hstack((torch.zeros_like(x), x))
    Uci = model(Sci)
    Uexp_ci = 1.0/(1.0 + torch.exp(torch.sqrt(lmbda/6.0)*x))**2
    loss_ci = torch.mean((Uci - Uexp_ci)**2)

    # condições de contorno

    # contorno x = 0
    Scc1 = torch.hstack((t, torch.zeros_like(t)))
    Ucc1 = model(Scc1)
    Ucc1_exp = 1.0/(1.0 + torch.exp(-5.0/6.0*lmbda*t))**2
    loss_cc1 = torch.mean((Ucc1 - Ucc1_exp)**2)

    # contorno x = 1
    Scc2 = torch.hstack((t, torch.ones_like(t)))
    Ucc2 = model(Scc2)
    Ucc2_exp = 1.0/(1.0 + torch.exp(torch.sqrt(lmbda/6.0) \
                    - 5.0/6.0*lmbda*t))**2
    loss_cc2 = torch.mean((Ucc2 - Ucc2_exp)**2)

    # dados
    Udata_est = model(S_data)
    loss_data = torch.mean((Udata_est - Us_data)**2)

    # função de perda total
    loss = loss_res + loss_ci + loss_cc1 + loss_cc2 + loss_data
    

    if (epoch % 100 == 0):
        print(f'{epoch}: lambda: {model.lmbda.item():.2e} loss_res: {loss.item():.2e} ci: {loss_ci.item():.1e} cc1: {loss_cc1.item():.1e} cc2: {loss_cc2.item():.1e}')
        torch.save(model, f'model.pt')

    # critério de parada
    if (epoch < window_size):
        lmbda_window.append(model.lmbda.item())
    else:
        lmbda_window.append(model.lmbda.item())
        lmbda_window.pop(0)
        lmbda_var = max(lmbda_window) - min(lmbda_window)

        if lmbda_var < window_tol and epoch > 5000:
            print(f'Critério de parada atingido na época {epoch}.')
            print(f'\tlambda: {model.lmbda.item():.2e}')
            break

    # backward
    optim.zero_grad()
    loss.backward()
    optim.step()
