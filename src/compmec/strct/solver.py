
import numpy as np
from numpy import linalg as la
from typing import Tuple

"""
In the end we have a linear system which is given by


"""

def solve(K : np.ndarray, F : np.ndarray, U:np.ndarray, TOLERANCE=1e-9) -> Tuple[np.ndarray, np.ndarray]:
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
    npts = F.shape[0]
    ndofs = F.shape[1]
    known = []
    unknown = []
    for i in range(npts):
        for j in range(ndofs):
            if U[i, j] is None:
                unknown.append((i,j))
            else:
                known.append((i,j))

    nu = len(unknown)
    nk = len(known)
    
    Kkk = np.zeros((nk, nk))
    Kku = np.zeros((nk, nu))
    Kuu = np.zeros((nu, nu))
    Fk = np.zeros(nu)
    Uk = np.zeros(nk)
    for i, (a, b) in enumerate(known):
        Uk[i] = U[a, b]
        for j, (c, d) in enumerate(known):
            Kkk[i, j] = K[a, b, c, d]
        for j, (c, d) in enumerate(unknown):
            Kku[i, j] = K[a, b, c, d]
    for i, (a, b)  in enumerate(unknown):
        Fk[i] = F[a, b]
        for j, (c, d) in enumerate(unknown):
            Kuu[i, j] = K[a, b, c, d]

    B = Fk - Kku.T @ Uk
    try:
        Uu = la.solve(Kuu, B)
    except np.linalg.LinAlgError as e:
        Uu = la.lstsq(Kuu, B, rcond=TOLERANCE)[0]
        
    Fu = Kkk @ Uk + Kku @ Uu

    
    for i, (a, b) in enumerate(known):
        F[a, b] += Fu[i]
    for i, (a, b) in enumerate(unknown):
        U[a, b] = Uu[i]
    U = np.array(U, dtype="float64")
    U[np.abs(U) < TOLERANCE] = 0
    
    return U, F




