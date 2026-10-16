from torch import tensor, sin, pi, ones_like

# entrada
x = tensor([0.,0.25,0.5,0.75,1.])
x = x.reshape(-1, 1)
x.requires_grad_(True)

# propagação
y = sin(pi * x)

# retropropagação (cálculo do gradiente)
y.backward(gradient=ones_like(y))

# print gradiente
print('dydx =', x.grad)

def print_backward_graph(node, depth=0, seen=None):
    seen = set() if seen is None else seen
    if node is None or node in seen:
        return
    seen.add(node)
    print("  " * depth + type(node).__name__)
    for parent, _ in node.next_functions:
        print_backward_graph(parent, depth + 1, seen)

print_backward_graph(y.grad_fn)

