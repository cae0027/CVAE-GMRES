from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

class KLFieldSampler:
    """
    Generates realizations of a 1D log-normal random field using 
    the Karhunen–Loève (KL) expansion with fixed eigen decomposition.
    """

    def __init__(self, mesh, sigma=1.0, ell=0.01, rougher=None):
        """
        Initialize the KL field sampler.

        Args:
            mesh (Mesh): 1D FEniCS mesh.
            sigma (float): Standard deviation of the underlying Gaussian field.
            ell (float): Correlation length (η) for the covariance kernel.
            rougher (callable): Optional rougheration applied before exponentiation.
        """
        self.mesh = mesh
        self.sigma = sigma
        self.eta = ell
        self.V = FunctionSpace(mesh, 'CG', 1)
        self.x = self.V.tabulate_dof_coordinates().flatten()
        self.n = len(self.x)
        self.rougher = rougher if rougher else lambda x: x
        self.evals, self.evecs = self._compute_kl_modes()

    def _kernel(self, xi, xj):
        return self.sigma**2 * np.exp(-((xi - xj) ** 2) / self.eta)

    def _compute_kl_modes(self):
        """Compute and store eigenvalues and eigenvectors of the covariance matrix."""
        C = np.fromfunction(
            lambda i, j: self._kernel(self.x[i.astype(int)], self.x[j.astype(int)]),
            shape=(self.n, self.n),
            dtype=float
        )
        evals, evecs = np.linalg.eigh(C)
        idx = np.argsort(evals)[::-1]
        evals = np.clip(evals[idx], a_min=0.0, a_max=None)
        evecs = evecs[:, idx]
        return evals, evecs


    def sample(self, z, mu=None):
        """
        Generate a random field sample using a provided z vector and optional spatial mean.

        Args:
            z (np.ndarray): Standard normal vector used in KL expansion.
            mu (Expression or Function, optional): Spatial mean function μ(x).

        Returns:
            Function: FEniCS Function representing the sampled log-normal field.
        """
        k = len(z)
        if k > len(self.evals):
            raise ValueError(f"Requested {k} modes, but only {len(self.evals)} available.")

        vals = np.sqrt(self.evals[:k]) * z
        G = self.evecs[:, :k] @ vals

        if mu is None:
            mu_vals = np.zeros(self.n)
        else:
            mu_interp = interpolate(mu, self.V)
            mu_vals = mu_interp.vector().get_local()

        log_q = mu_vals + G
        q_vals = np.exp(self.rougher(log_q))

        q = Function(self.V)
        q.vector().set_local(q_vals)
        q.vector().apply("insert")
        return q
    
    

    def compute_local_stiffness(self, q_func, x_i, x_ip1):
        submesh = IntervalMesh(1, x_i, x_ip1)
        V_sub = FunctionSpace(submesh, "CG", 1)
        u, v = TrialFunction(V_sub), TestFunction(V_sub)

        q_expr = Func2Expr(q_func, degree=4)
        a = q_expr * dot(grad(u), grad(v)) * dx
        A_local = assemble(a).array()
        return A_local[0, 0], A_local[0, 1], A_local[1, 1]



class Func2Expr(UserExpression):
    """
    Wrap a FEniCS Function as a UserExpression for mesh-independent evaluation.
    Useful for evaluating a Function on a different mesh.
    """

    def __init__(self, fenics_function, **kwargs):
        super().__init__(**kwargs)
        self._func = fenics_function

    def eval(self, values, x):
        values[0] = self._func(Point(x[0]))

    def value_shape(self):
        return ()



# ---- Example Usage ----
if __name__ == '__main__':
    N = 1000
    mesh = IntervalMesh(N, -1.0, 1.0)
    samples = 15
    max_q = 0.0

    def rough_fn(x): return np.cos(x / 0.5) * x

    mu = Expression("1 + 0.3*sin(pi*x[0])", degree=2)

    sampler = KLFieldSampler(mesh, ell=0.005, rougher=None)

    for _ in range(samples):
        z = np.random.randn(10)
        q = sampler.sample(z, mu=mu)
        q_vals = q.vector().get_local()
        max_q = max(max_q, np.max(q_vals))
        plt.ylim(0, max_q * 1.05)
        plot(q)

    plt.xlabel("x")
    plt.ylabel("q(x)")
    plt.title("KL Log-Normal Random Fields with Spatial Mean")
    plt.show()