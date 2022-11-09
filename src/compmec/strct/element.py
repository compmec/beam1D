
"""
Each point has 6 unknowns:

"""
import numpy as np
from scipy import sparse
from compmec.strct.__classes__ import Structural1D
from compmec.strct.fields import ComputeFieldBeam

def compute_rvw(p0: tuple, p1: tuple) -> np.ndarray:
    np0 = np.zeros(3)
    np1 = np.zeros(3)
    np0[:len(p0)] = p0
    np1[:len(p1)] = p1
    dp = np.array(np1 - np0)
    L = np.linalg.norm(dp)
    r = dp/L
    v = (0, 0, 1)
    cosangle = np.inner(r, v)
    if np.abs(cosangle) > 0.99:  # 0.99 is the cos of 8 degrees
        v = (0, 1, 0)
        cosangle = np.inner(r, v)
    v -= cosangle * r
    v /= np.linalg.norm(v)
    w = np.cross(v, r)
    R = np.zeros((3, 3))
    R[:, 0] = r
    R[:, 1] = w
    R[:, 2] = v
    return R.T


class Truss(Structural1D):
    def __init__(self, path):
        self._dofs = 3
        super().__init__(path)
        
class Cable(Structural1D):
    def __init__(self, path):
        self._dofs = 3
        super().__init__(path)

class Beam(Structural1D):
    def __init__(self, path):
        self._dofs = 6
        self.__result = None
        super().__init__(path)

    def local_stiffness_matrix(self, p0: tuple, p1: tuple) -> np.ndarray:
        raise NotImplementedError("This function must be overwritten by the child")
    
    def global_stiffness_matrix(self, p0: tuple, p1: tuple) -> np.ndarray:
        Kloc = self.local_stiffness_matrix(p0, p1)
        R33 = compute_rvw(p0, p1)
        Kglo = np.zeros((2, 6, 2, 6), dtype="float64")
        for i in range(2):
            for j in range(2):
                Kglo[i, :3, j, :3] = R33.T @ Kloc[i, :3, j, :3] @ R33
                Kglo[i, :3, j, 3:] = R33.T @ Kloc[i, :3, j, 3:] @ R33
                Kglo[i, 3:, j, :3] = R33.T @ Kloc[i, 3:, j, :3] @ R33
                Kglo[i, 3:, j, 3:] = R33.T @ Kloc[i, 3:, j, 3:] @ R33
        return Kglo

    def stiffness_matrix(self) -> np.ndarray:
        points = [self.path(ti) for ti in self.ts]
        npts = len(points)
        Kglobal = np.zeros((npts, 6, npts, 6))
        for i in range(npts-1):
            p0, p1 = points[i], points[i+1]
            Kgloone = self.global_stiffness_matrix(p0, p1)
            Kglobal[i:i+2, :, i:i+2, :] += Kgloone
        return Kglobal

    @property
    def result(self):
        if self.__result is None:
            raise ValueError("You must run the simulation to get the results")
        return self.__result

    @result.setter
    def result(self, value: np.ndarray):
        n = self.path.P.shape[0]  # Number of points in the beam
        value = np.array(value)
        if value.ndim != 2:
            raise ValueError("The received result must be 2D matrix")
        npts, dim = value.shape
        if dim != 6:
            raise ValueError(f"The received dimension of result must be {6}, not {dim}")
        if npts != n:
            raise ValueError(f"The number of points is {n}, not {npts}")
        self.__result = value

    def field(self, fieldname: str):
        if not isinstance(fieldname, str):
            raise TypeError("The fieldname must be a string like 'L2norm(u)'")
        return ComputeFieldBeam.field(fieldname, self, self.result)


class EulerBernoulli(Beam):
    def __init__(self, path):
        super().__init__(path)

    def local_stiffness_matrix_Kx(self, L: float) -> np.ndarray:
        E = self.material.E
        A = self.section.Ax 
        Kx = (E*A/L) * (2*np.eye(2)-1) 
        return Kx 
 
    def local_stiffness_matrix_Kt(self, L: float) -> np.ndarray: 
        G = self.material.G
        Ix = self.section.Ix
        Kt = (G*Ix/L) * (2*np.eye(2)-1) 
        return Kt 
         
    def local_stiffness_matrix_Ky(self, L: float) -> np.ndarray: 
        E = self.material.E
        Iz = self.section.Iz
        Ky = np.array([[ 12,    6*L,  -12,    6*L], 
                       [6*L, 4*L**2, -6*L, 2*L**2], 
                       [-12,   -6*L,   12,   -6*L], 
                       [6*L, 2*L**2, -6*L, 4*L**2]]) 
        return (E*Iz/L**3) * Ky 
 
    def local_stiffness_matrix_Kz(self, L: float) -> np.ndarray: 
        E = self.material.E
        Iy = self.section.Iy 
        Kz = np.array([[  12,   -6*L,  -12,   -6*L], 
                       [-6*L, 4*L**2,  6*L, 2*L**2], 
                       [ -12,    6*L,   12,    6*L], 
                       [-6*L, 2*L**2,  6*L, 4*L**2]]) 
        return (E*Iy/L**3) * Kz 

    def local_stiffness_matrix(self, p0: tuple, p1: tuple) -> np.ndarray:
        """
        With two points we will have a matrix [12 x 12]
        But we are going to divide the matrix into [x, y, z] coordinates
        That means, our matrix is in fact [4, 3, 4, 3]
        Or also  [2, 6, 2, 6]
        """
        L = np.linalg.norm(np.array(p1)-np.array(p0))
        Kx = self.local_stiffness_matrix_Kx(L)
        Kt = self.local_stiffness_matrix_Kt(L)
        Ky = self.local_stiffness_matrix_Ky(L)
        Kz = self.local_stiffness_matrix_Kz(L)
        K = np.zeros((2, 6, 2, 6))
        K[:, 0, :, 0] = Kx
        K[:, 3, :, 3] = Kt
        for i in range(2):
            for j in range(2):
                for wa, a in enumerate([1, 5]):
                    for wb, b in enumerate([1, 5]):
                        K[i, a, j, b] = Ky[2*i+wa, 2*j+wb]
                for wa, a in enumerate([2, 4]):
                    for wb, b in enumerate([2, 4]):
                        K[i, a, j, b] = Kz[2*i+wa, 2*j+wb]
        return K

    


class Timoshenko(Beam):
    def __init__(self, path):
        super().__init__(path)



