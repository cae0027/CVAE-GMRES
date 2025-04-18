from dolfin import *
from scipy.sparse.linalg import gmres, LinearOperator, spilu
import numpy as np
import matplotlib.pyplot as plt

from helpers import reconsStiffness

def apply_boundary_conditions(A, b, boundary_dofs):
    """ Apply Dirichlet boundary conditions by removing rows/columns. """
    interior_dofs = np.setdiff1d(np.arange(A.shape[0]), boundary_dofs)
    A_bc = A[np.ix_(interior_dofs, interior_dofs)]
    b_bc = b[interior_dofs]
    return A_bc, b_bc, interior_dofs
    


def testAoAlMatrices(Al, Ao):
    N = len(Al) - 1
    a, b = -1.0, 1.0
    mesh = IntervalMesh(N, a, b)
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Assemble RHS vector
    v = TestFunction(V)
    f = Constant(1.0)
    L = inner(f, v) * dx
    b_vec = assemble(L)

    # Identify boundary DOFs explicitly
    boundary_dofs = np.array([0, V.dim()-1])

    # Convert FEniCS vector to NumPy array
    b_np = b_vec.get_local()

    # Apply boundary conditions explicitly to Al and Ao
    Al_bc, b_bc, interior_dofs = apply_boundary_conditions(Al, b_np, boundary_dofs)
    Ao_bc, _, _ = apply_boundary_conditions(Ao, b_np, boundary_dofs)
    min_diag = np.min(np.abs(np.diagonal(Al_bc)))
    print("Minimum absolute diagonal entry in Al_bc:", min_diag)


    # Solve with learned matrix
    ilu_l = spilu(Al_bc)
    Ml = LinearOperator(Al_bc.shape, ilu_l.solve)
    ul_interior, info_l = gmres(Al_bc, b_bc, M=Ml)
    if info_l > 0:
        print(f"[Learned] GMRES did not converge (info = {info_l})")

    # Solve with original matrix
    ilu_o = spilu(Ao_bc)
    Mo = LinearOperator(Ao_bc.shape, ilu_o.solve)
    uo_interior, info_o = gmres(Ao_bc, b_bc, M=Mo)
    if info_o > 0:
        print(f"[Original] GMRES did not converge (info = {info_o})")

    # Reconstruct full solution vectors including boundary conditions
    ul_full = np.zeros(V.dim())
    uo_full = np.zeros(V.dim())
    ul_full[interior_dofs] = ul_interior
    uo_full[interior_dofs] = uo_interior

    # Convert back to FEniCS functions
    ul = Function(V)
    ul.vector().set_local(ul_full)
    uo = Function(V)
    uo.vector().set_local(uo_full)

    # Compute L2 error
    error_L2 = errornorm(uo, ul, norm_type='L2')
    print(f"L2 error between original and learned: {error_L2:.6e}")

    # Plot solutions
    plt.figure()
    plot(uo, label="Original")
    plot(ul, label="Learned", linestyle="--")
    plt.legend()
    plt.title("Solution Comparison")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid(True)
    plt.show()

    return ul, uo, error_L2


if __name__ == "__main__":
    # Load Al and Ao
    Al = np.load('Apred.npy', allow_pickle=False).astype(np.float64)
    Ao = np.load('Af.npy', allow_pickle=False).astype(np.float64)

    Al = reconsStiffness(Al[:1001], Al[1001:])
    Ao = reconsStiffness(Ao[:1001], Ao[1001:])

    # Run the modified test
    ul, uo, error_L2 = testAoAlMatrices(Al, Ao)
