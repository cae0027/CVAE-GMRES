from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import LinearOperator, spilu


from helpers import vecStiffness, reconsStiffness

def getVecStiffness(N,a,b,xi):
    """
    Assemble stiffness matrix for random field sample and return vectorized nonzero entries.
    
    Args:
        N: Number of elements
        a: Left boundary
        b: Right boundary
        xi: uniform random variable in [0, 2pi]

    Returns:
        vectorized main and upper diagonal entries of the stiffness matrix (length 2N + 1)
    """
    mesh = IntervalMesh(N, a,b)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    q_fn = Expression("exp(5*cos(2*pi*x[0]/0.1 + xi))", degree=4, xi=xi)
    q_fn = interpolate(q_fn, V)
    # # plot q_fn
    # plot(q_fn)
    # plt.show()

    af = inner(q_fn * u.dx(0), v.dx(0)) * dx
    A = assemble(af)
    # define dirichlet boundary conditions
    bc = DirichletBC(V, Constant(0.0), "on_boundary")
    # apply BCs to A
    bc.apply(A)

    ############# Test that the generated stidffness matrix is correct #############
    # # solve with constant right hand side
    # bf = Constant(1.0)
    # L = inner(bf, v) * dx
    # sol1 = Function(V)
    # sol2 = Function(V)
    # # solve
    # solve(A,sol1.vector(), assemble(L))
    # solve(af==L, sol2, bc)
    # # plot the solution
    # plot(sol1,label="Ax=b")
    # plot(sol2,label="var form")
    # plt.legend()
    # plt.show()
    ################################### End of test ################################

    return vecStiffness(A)


if __name__ == "__main__":
    N = 1000  # mesh elements
    a, b = -1.0, 1.0
    m = 5 # number of samples
    xi = np.random.uniform(0, 2*np.pi, m)

    # Define or interpolate eigenfunctions into the function space V (1D Interval)
    mesh = IntervalMesh(N, a, b)
    V = FunctionSpace(mesh, "Lagrange", 1)

    for i in range(m):
        vec_K = getVecStiffness(N,a,b,xi[i])
        print(vec_K.shape)

        # Reconstruct the stiffness matrix
        A = reconsStiffness(vec_K[:N+1], vec_K[N+1:])
        print(A.shape)

        # solve with constant right hand side using GMRES
        v = TestFunction(V)
        bf = Constant(1.0)
        L = inner(bf, v) * dx
        L = assemble(L)
        bc = DirichletBC(V, Constant(0.0), "on_boundary")
        bc.apply(L)
        # apply ILU preconditioner
        ilu = spilu(A)
        M = LinearOperator(A.shape, ilu.solve)
        uvec, info = gmres(A, L, M=M)
        if info > 0:
            print("GMRES did not converge, info = ", info)
        # plot the solution
        u = Function(V)
        u.vector()[:] = uvec
        plot(u)
    plt.show()

