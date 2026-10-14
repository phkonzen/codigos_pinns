# codigos_pinns
Códigos exemplos do livro **Introdução a PINNs**

## Exemplo: problema de contorno em 1D

O script [`examples/ode_pinn.py`](examples/ode_pinn.py) usa uma rede neural
para aproximar a solução do problema

```text
u''(x) + π² sin(πx) = 0,  0 < x < 1
u(0) = u(1) = 0
```

A solução exata é `u(x) = sin(πx)`. O exemplo calcula o resíduo da equação
com diferenciação automática e combina seu erro com o erro nas condições de
contorno para treinar a rede.

Com Python 3.9 ou superior, instale a dependência e execute o exemplo:

```bash
python -m pip install -r requirements.txt
python examples/ode_pinn.py
```

O programa informa a perda final e o erro máximo em uma malha de avaliação.
