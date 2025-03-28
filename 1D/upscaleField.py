import numpy as np
import fenics as fe
import matplotlib.pyplot as plt


class RandomFieldUpscaler:
    def __init__(self, d, c, beta, domain=(0, 1), fine_res=1000, coarse_res=10):
        self.d = d
        self.c = c
        self.beta = beta
        self.domain = domain
        self.fine_res = fine_res
        self.coarse_res = coarse_res
        self.x_fine = np.linspace(domain[0], domain[1], fine_res)

    def q_fine(self, xi):
        return self.d + np.exp(self.c * np.cos(2 * np.pi * self.x_fine / self.beta + xi))

    def local_average(self, q):
        bins = np.array_split(q, self.coarse_res)
        return np.array([np.mean(b) for b in bins])

    def projection_fem(self, xi):
        mesh = fe.IntervalMesh(self.coarse_res, *self.domain)
        V = fe.FunctionSpace(mesh, 'P', 1)
        q_expr = fe.Expression(
            'd + exp(c * cos(2*pi*x[0]/beta + xi))',
            degree=4, d=self.d, c=self.c, beta=self.beta,
            xi=float(xi), pi=np.pi
        )
        q_func = fe.interpolate(q_expr, V)
        q_proj = q_func.compute_vertex_values(mesh)
        dof_coords = V.tabulate_dof_coordinates().flatten()
        sorted_indices = np.argsort(dof_coords)
        return q_proj[sorted_indices], np.sort(dof_coords)

    def truncated_fourier(self, q, n_modes=5):
        fft = np.fft.fft(q)
        fft_copy = np.zeros_like(fft)
        fft_copy[:n_modes] = fft[:n_modes]
        fft_copy[-n_modes + 1:] = fft[-n_modes + 1:]
        q_smooth = np.fft.ifft(fft_copy).real
        return self.local_average(q_smooth)

    def homogenized(self, xi):
        coarse_edges = np.linspace(*self.domain, self.coarse_res + 1)
        q_sample = self.q_fine(xi)
        q_hom = np.zeros(self.coarse_res)

        for i in range(self.coarse_res):
            mask = (self.x_fine >= coarse_edges[i]) & (self.x_fine < coarse_edges[i + 1])
            q_cell = q_sample[mask]
            dx = self.x_fine[1] - self.x_fine[0]
            q_hom[i] = (1.0 / np.trapz(1.0 / q_cell, dx=dx)) * (coarse_edges[i + 1] - coarse_edges[i])

        return q_hom

    def upscale(self, xi, method="local_average", **kwargs):
        q_high = self.q_fine(xi)
        x_fine = self.x_fine

        if method == "local_average":
            q_coarse = self.local_average(q_high)
            x_coarse = 0.5 * (np.linspace(*self.domain, self.coarse_res + 1)[:-1] +
                              np.linspace(*self.domain, self.coarse_res + 1)[1:])

        elif method == "projection_fem":
            q_coarse, x_coarse = self.projection_fem(xi)

        elif method == "fourier":
            n_modes = kwargs.get("n_modes", 5)
            q_coarse = self.truncated_fourier(q_high, n_modes=n_modes)
            x_coarse = 0.5 * (np.linspace(*self.domain, self.coarse_res + 1)[:-1] +
                              np.linspace(*self.domain, self.coarse_res + 1)[1:])

        elif method == "homogenized":
            q_coarse = self.homogenized(xi)
            x_coarse = 0.5 * (np.linspace(*self.domain, self.coarse_res + 1)[:-1] +
                              np.linspace(*self.domain, self.coarse_res + 1)[1:])

        else:
            raise ValueError(f"Unsupported method: {method}")

        return q_coarse, q_high, x_coarse, x_fine


if __name__ == "__main__":
    domain = (-1,1)
    up = RandomFieldUpscaler(d=1.0, c=2.0, beta=0.1, coarse_res=500, domain=domain)
    xi = np.random.uniform(0, 2 * np.pi)
    
    # Choose one of: "local_average", "projection_fem", "fourier", "homogenized", "sample_based"
    method = "local_average"
    q_coarse, q_fine, x_coarse, x_fine = up.upscale(xi, method=method, n_modes=100)

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(x_fine, q_fine, label="Fine-scale $q(x, \\xi)$", lw=1.5)
    plt.plot(x_coarse, q_coarse, "o-", label=f"Upscaled ({method})", lw=2)
    plt.xlabel("x")
    plt.ylabel("q(x)")
    plt.title(f"Upscaling Method: {method}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
