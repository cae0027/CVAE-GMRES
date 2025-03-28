# save_data_numpy.py
import numpy as np
from generateA import getVecStiffness
from upscaleField import RandomFieldUpscaler

# set seed for reproducibility
np.random.seed(54)

def generate_and_save_data(
    save_path='data.npz',
    m=50,
    a=-1,
    b=1,
    nc=100,
    nf=1000,
    d=0,
    c=5,
    beta=0.1,
    upsType='local_average',
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    xi = np.random.uniform(0, 2 * np.pi, m)
    qc = np.empty((m, nc))
    A = np.empty((m, 2 * nf + 1))

    for i in range(m):
        rf_up = RandomFieldUpscaler(d, c, beta, coarse_res=nc, domain=(a, b))
        qc[i] = rf_up.upscale(xi[i], method=upsType)[0]
        A[i] = getVecStiffness(nf, a, b, xi[i])

    np.savez_compressed(save_path, A=A, qc=qc)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    generate_and_save_data(save_path="fenics_output.npz", m=1000)
