import torch

# dispositivo GPU/CPU ?
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

# discretização temporal
t0 = 0.0 # tempo inicial
tf = 1.0 # tempo final
nt = 10 # num de passos no tempo
ts = torch.linspace(t0, tf, nt+1, device=device) # instantes de tempo
ht = (tf - t0)/nt # tamanho do passo no tempo

# modelo
n_h = 2 # num de camadas escondidas
n_n = 25 # num de neurônios por camada escondida
act_fun = torch.nn.Tanh() # função de ativação

model = torch.nn.Sequential()
# camada de entrada
model.add_module('layer_1', torch.nn.Linear(1,n_n))
model.add_module('fun_1', act_fun)
# camadas escondidas
for i in range(n_h-1):
    model.add_module(f'layer_{i+2}', torch.nn.Linear(n_n,n_n))
    model.add_module(f'fun_{i+2}', act_fun)
# camada de saída
model.add_module(f'layer_{n_h+1}', torch.nn.Linear(n_n,nt+1))
model.to(device) # envia modelo para o dispositivo

# treinamento

## fonte
def f(t, x):
    return (torch.pi**2 - 1) * torch.exp(-t) * torch.sin(torch.pi*x)

## optimizador
optim = torch.optim.Adam(model.parameters(), lr=0.001)

## num de amostras por época
ns_x = 40 # num de amostras no espaço

## num max épocas
nepochs = 50_000

## tolerância
tol = 1e-4
persist = 5

# contador de persistência
n_conv = 0

# laço de treinamento
for epoch in range(nepochs):

    model.train()

    # amostras de pontos internos
    Xs = torch.rand(ns_x, 1, device=device)
    Xs.requires_grad = True
    
    # propagação nos pontos internos
    U = model(Xs)

    loss_res = 0.0
    for n in range(0, nt):

        # derivadas de primeira ordem
        DUDX = torch.autograd.grad(U[:,n+1:n+2], Xs,
                            grad_outputs=torch.ones_like(U[:,n+1:n+2]),
                            create_graph=True)[0]

        # derivadas de segunda ordem
        D2UDX = torch.autograd.grad(DUDX, Xs,
                            grad_outputs=torch.ones_like(DUDX),
                            create_graph=True)[0]

        # resíduo da equação discretizada (Euler implícito)
        R = U[:,n:n+1] + ht * f(ts[n+1], Xs) - U[:,n+1:n+2] + ht * D2UDX

        loss_res += torch.sum(R**2)

    # função de perda: resíduo
    loss_res = loss_res / (ns_x * nt)

    # condição inicial
    Uexp_ci = torch.sin(torch.pi*Xs)
    loss_ci = torch.mean((U[:,0:1] - Uexp_ci)**2)

    # condições de contorno

    # contorno x = 0
    Ucc1 = model(torch.zeros((1,1), device=device))
    loss_cc1 = torch.mean((Ucc1 - 0.0)**2)

    # contorno x = 1
    Ucc2 = model(torch.ones((1,1), device=device))
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
