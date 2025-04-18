from dolfin import *
import numpy as np
import json
from tqdm import trange
from klField import KLFieldSampler  # must include compute_local_stiffness inside
import os

# -----------------------------
# Parameters
# -----------------------------
a, b = -1.0, 1.0          # Domain bounds
nf = 1000                 # Number of elements in the global mesh
num_samples = 100          # Number of KL field realizations
nof_kl = 10             # Number of KL modes to use
ell = 0.005
qp = 20 # No of quad points for data generation only
save_path = "fem_stiffness_data_kl.json"

mesh = IntervalMesh(nf, a, b)
x_nodes = np.linspace(a, b, nf + 1)

mu = Expression("1 + 0.3*sin(pi*x[0])", degree=2) 
sampler = KLFieldSampler(mesh, ell=ell, rougher=None)

dataset = []
for _ in trange(num_samples, desc="Generating KL samples"):
    seed = np.random.randint(0, 2**32 - 1)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(nof_kl)
    # Generate a sample q(x)
    q_func = sampler.sample(z, mu=mu)
    element_data = []
    for i in range(nf):
        x_i, x_ip1 = x_nodes[i], x_nodes[i + 1]
        A_ii, A_ij, A_jj = sampler.compute_local_stiffness(q_func, x_i, x_ip1)
        # sample points btw x_i and x_ip1 for q(x)
        xq = np.linspace(x_i, x_ip1, qp)
        q_vals = [q_func(Point(x)) for x in xq]
        element_data.append({
            "interval": [x_i, x_ip1],
            "q_vals": q_vals,
            "stiffness": [A_ii, A_ij, A_jj]
        })

    dataset.append({
        "seed": seed,
        "elements": element_data
    })

# -----------------------------
# Save Dataset
# -----------------------------
# os.makedirs(os.path.dirname(save_path), exist_ok=True)

with open(save_path, "w") as f:
    json.dump(dataset, f)

print(f"✅ Dataset saved to: {save_path}")
