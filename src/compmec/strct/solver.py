from typing import Tuple

import numpy as np
from numpy import linalg as la

"""
In the end we have a linear system which is given by


"""


def solve(
    K: np.ndarray, F: np.ndarray, U: np.ndarray, TOLERANCE=1e-9
) -> Tuple[np.ndarray, np.ndarray]:
    """
    K is a big matrix of shape (npts, 6, npts, 6)
    F is a matrix of shape (npts, 6)
    U is a matrix of the values of U, of shape (npts, 6)
    That means, U is like
    U = [[1, None, 0, None, None, None],
         [None, 0, None, None, None, None],
         ...
         []]
    """
    npts, ndofs = U.shape
    Kexp = K.reshape((npts * ndofs, npts * ndofs))
    Fexp = F.reshape((npts * ndofs)).astype("float64")
    Uexp = U.reshape((npts * ndofs))
    mask = Uexp == None
    Uk = np.delete(Uexp, mask).astype("float64")
    Fk = np.delete(Fexp, ~mask)
    Kkk = np.delete(np.delete(Kexp, mask, axis=0), mask, axis=1)
    Kku = np.delete(np.delete(Kexp, mask, axis=0), ~mask, axis=1)
    Kuu = np.delete(np.delete(Kexp, ~mask, axis=0), ~mask, axis=1)

    B = Fk - Kku.T @ Uk
    try:
        Uu = la.solve(Kuu, B)
    except np.linalg.LinAlgError:
        Uu = la.lstsq(Kuu, B, rcond=TOLERANCE)[0]
    Uu[np.abs(Uu) < TOLERANCE] = 0
    Uexp[mask] = Uu
    Fu = Kkk @ Uk + Kku @ Uu
    Fexp[~mask] += Fu
    F = Fexp.reshape((npts, ndofs))
    U = Uexp.reshape((npts, ndofs)).astype("float64")
    return U, F
