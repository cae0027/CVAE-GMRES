from dolfin import *
from scipy.sparse.linalg import gmres, LinearOperator, spilu
import numpy as np
import matplotlib.pyplot as plt

from helpers import reconsStiffness

def testAoAlMatrices(Al, Ao):
    """
    Compare PDE solutions using learned and original stiffness matrices.

    Parameters:
    - Al: learned stiffness matrix (SciPy CSR)
    - Ao: original stiffness matrix (SciPy CSR)
    """
    N = len(Al) - 1
    a, b = -1.0, 1.0
    mesh = IntervalMesh(N, a, b)
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Assemble RHS vector with constant forcing
    v = TestFunction(V)
    f = Constant(1.0)
    L = inner(f, v) * dx
    b_vec = assemble(L)

    # Apply homogeneous Dirichlet BC
    bc = DirichletBC(V, Constant(0.0), "on_boundary")
    bc.apply(b_vec)

    # Solve with learned matrix
    ilu_l = spilu(Al)
    Ml = LinearOperator(Al.shape, ilu_l.solve)
    ul_vec, info_l = gmres(Al, b_vec.copy(), M=Ml)
    if info_l > 0:
        print(f"[Learned] GMRES did not converge (info = {info_l})")

    # Solve with original matrix
    ilu_o = spilu(Ao)
    Mo = LinearOperator(Ao.shape, ilu_o.solve)
    uo_vec, info_o = gmres(Ao, b_vec.copy(), M=Mo)
    if info_o > 0:
        print(f"[Original] GMRES did not converge (info = {info_o})")

    # Convert to FEniCS functions
    ul = Function(V)
    ul.vector()[:] = ul_vec
    uo = Function(V)
    uo.vector()[:] = uo_vec

    # L2 error
    error_L2 = errornorm(uo, ul, norm_type='L2')
    print(f"L2 error between original and learned: {error_L2:.6e}")

    # Plot comparison
    plt.figure()
    # get min and max of uo, ul
    max_val = max(np.max(uo.vector()[:]), np.max(ul.vector()[:]))
    min_val = min(np.min(uo.vector()[:]), np.min(ul.vector()[:])
    )
    min_val, max_val = min_val-0.1*abs(min_val), max_val+0.05*abs(max_val)
    plot(uo, label="Original")
    plot(ul, label="Learned", linestyle="--")
    plt.legend()
    plt.title("Solution Comparison")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid(True)
    # plt.ylim([min_val, max_val])
    # save the plot
    # plt.savefig('./results/img50.png')
    plt.show()

    return ul, uo, error_L2

if __name__ == "__main__":
    # load Ao and Al
    Al = np.load('Apred.npy', allow_pickle=False).astype(np.float64)
    Ao = np.load('Af.npy', allow_pickle=False).astype(np.float64)
    print(Al.shape, Ao.shape)
    Al = reconsStiffness(Al[:1001], Al[1001:])
    Ao = reconsStiffness(Ao[:1001], Ao[1001:])
    # plot diagonal of Ao
    plt.plot(Ao.diagonal())
    plt.show()
    # testAoAlMatrices(Al, Ao)
    # print(Al)
    