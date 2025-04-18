from fenics import as_backend_type
import numpy as np
from scipy.sparse import diags

def vecStiffness(A):
    """
    Efficiently extract the main diagonal and upper diagonal of a symmetric, tridiagonal matrix.

    Args:
        A: FEniCS PETScMatrix (symmetric, tridiagonal)

    Returns:
        diag:     (N,) main diagonal entries
        upper:    (N-1,) upper diagonal entries
    """
    A_petsc = as_backend_type(A).mat()
    csr_indptr, csr_indices, csr_data = A_petsc.getValuesCSR()
    N = A_petsc.getSize()[0]

    # Prepare arrays
    diag = np.zeros(N)
    upper = np.zeros(N - 1)

    for i in range(N):
        row_start = csr_indptr[i]
        row_end = csr_indptr[i + 1]
        cols = csr_indices[row_start:row_end]
        vals = csr_data[row_start:row_end]

        for col, val in zip(cols, vals):
            if col == i:
                diag[i] = val
            elif col == i + 1:
                upper[i] = val
            # Skip col < i (lower triangle) and col > i+1 (non-tridiagonal)

    return np.concatenate((diag, upper))


def reconsStiffness(diag, upper):
    """
    Reconstruct full symmetric tridiagonal matrix from vectorized form.

    Args:
        diag:  (N,) main diagonal
        upper: (N-1,) upper diagonal

    Returns:
        scipy.sparse.csr_matrix of shape (N, N)
    """
    return diags(
        diagonals=[upper, diag, upper],
        offsets=[1, 0, -1],
        format="csr"
    )
