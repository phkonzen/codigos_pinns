"""PINN for the 1D boundary-value problem u'' + pi^2 sin(pi*x) = 0."""

import argparse

import torch
from torch import nn


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


def train(epochs, seed):
    torch.manual_seed(seed)
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    boundary_x = torch.tensor([[0.0], [1.0]])
    boundary_u = torch.zeros_like(boundary_x)

    for _ in range(epochs):
        x = torch.rand(128, 1, requires_grad=True)
        u = model(x)
        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0]
        residual = u_xx + torch.pi**2 * torch.sin(torch.pi * x)
        boundary_error = model(boundary_x) - boundary_u
        loss = residual.square().mean() + 10 * boundary_error.square().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model, loss.item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    model, final_loss = train(args.epochs, args.seed)
    x = torch.linspace(0, 1, 101).reshape(-1, 1)
    with torch.no_grad():
        error = (model(x) - torch.sin(torch.pi * x)).abs().max().item()

    print(f"Perda final: {final_loss:.6g}")
    print(f"Erro máximo: {error:.6g}")


if __name__ == "__main__":
    main()
