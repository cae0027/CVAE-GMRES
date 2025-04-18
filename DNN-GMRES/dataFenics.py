from dolfin import *
import numpy as np
import json
from tqdm import trange

# ---------------------------------------
# Parameters
# ---------------------------------------
a,b = (-1.0, 1.0)
nf = 1000  # Number of elements
m = 10  # Number of random field realizations
save_path = "fem_local_stiffness_data.json"

# ---------------------------------------
# Random Field Function q(x, ξ)
# ---------------------------------------
def create_q_expression(xi, d=0.0, c=5.0, beta=0.1):
    return Expression("d + exp(c * cos(2*pi*x[0]/beta + xi))",
                      degree=4, d=d, c=c, beta=beta, xi=xi, pi=np.pi)

# ---------------------------------------
# Compute Local Stiffness Over [x_i, x_{i+1}]
# ---------------------------------------
def compute_local_stiffness(x_i, x_ip1, q_expr):
    mesh = IntervalMesh(1, x_i, x_ip1)
    V = FunctionSpace(mesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)
    a = q_expr * dot(grad(u), grad(v)) * dx
    A_local = assemble(a).array()
    return A_local[0, 0], A_local[0, 1], A_local[1, 1]

# ---------------------------------------
# Main Data Generator
# ---------------------------------------
x_nodes = np.linspace(a,b, nf + 1)
dataset = []

for _ in trange(m, desc="Generating samples"):
    xi = float(np.random.uniform(0, 2 * np.pi))
    q_expr = create_q_expression(xi)
    element_data = []

    for i in range(nf):
        x_i, x_ip1 = x_nodes[i], x_nodes[i + 1]
        A_ii, A_ij, A_jj = compute_local_stiffness(x_i, x_ip1, q_expr)
        element_data.append({
            "interval": [x_i, x_ip1],
            "stiffness": [A_ii, A_ij, A_jj]
        })

    dataset.append({
        "xi": xi,
        "elements": element_data
    })

# ---------------------------------------
# Save to JSON
# ---------------------------------------
# with open(save_path, "w") as f:
#     json.dump(dataset, f)

# print(f"Saved dataset to {save_path}")
