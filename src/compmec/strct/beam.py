
"""
Each point has 6 unknowns:

"""
import numpy as np
from compmec.strct.__classes__ import Structural1D

class Truss(Structural1D):
    def __init__(self, path):
        super().__init__(path)


class Cable(Structural1D):
    def __init__(self, path):
        super().__init__(path)

class Beam(Structural1D):
    cos8 = 0.99
    def __init__(self, path):
        """
        vertical is the perpendicular direction to the line
        ver
        """
        super().__init__(path)
        self.set_random_v()
        
    @property
    def r(self) -> np.ndarray:
        return self._p/self.L

    @property
    def v(self) -> np.ndarray:
        return self._v

    @property
    def w(self) -> np.ndarray:
        return self._w

    @v.setter
    def v(self, value: np.ndarray) -> None:
        value = np.array(value, dtype="float64")
        value /= np.sqrt(np.sum(value**2))
        Lcostheta = np.inner(self.p, value) 
        if np.abs(Lcostheta) > self.L * Beam.cos8:
            raise ValueError("The received vector v must not be colinear to p")
        self._v = value - Lcostheta * self.p/(self.L**2)
        self._v /= np.linalg.norm(self._v)
        self._w = np.cross(self._v, self.r)

    def set_random_v(self) -> np.ndarray:
        if np.abs(self.p[2]) < self.L * Beam.cos8:
            self.v = (0, 0, 1)
        else:
            v = np.random.rand(3)
            self.v = v - np.inner(v, self.p) * self.p / (self.L**2)
            
    def rotation_matrix33(self) -> np.ndarray:
        return np.array([self.r, self.w, self.v])
            
    def global_stiffness_matrix(self) -> np.ndarray:
        Kloc = self.local_stiffness_matrix()
        R33 = self.rotation_matrix33()
        Kglo = np.zeros((2, 6, 2, 6), dtype="float64")
        for i in range(2):
            for j in range(2):
                Kglo[i, :3, j, :3] = R33.T @ Kloc[i, :3, j, :3] @ R33
                Kglo[i, :3, j, 3:] = R33.T @ Kloc[i, :3, j, 3:] @ R33
                Kglo[i, 3:, j, :3] = R33.T @ Kloc[i, 3:, j, :3] @ R33
                Kglo[i, 3:, j, 3:] = R33.T @ Kloc[i, 3:, j, 3:] @ R33
        return Kglo

class EulerBernoulli(Beam):
    def __init__(self, path):
        super().__init__(path)

    def stiffness_matrix(self) -> np.ndarray:
        return self.global_stiffness_matrix()

    def local_stiffness_matrix(self) -> np.ndarray:
        """
        With two points we will have a matrix [12 x 12]
        But we are going to divide the matrix into [x, y, z] coordinates
        That means, our matrix is in fact [4, 3, 4, 3]
        Or also  [2, 6, 2, 6]
        """
        L = self.L
        K = np.zeros((2, 6, 2, 6))
        E = self.material.E
        G = self.material.G
        A = self.section.Ax
        Ix = self.section.Ix
        Iy = self.section.Iy
        Iz = self.section.Iz

        Kx = (E*A/L) * (2*np.eye(2)-1)
        Kt = (G*Ix/L) * (2*np.eye(2)-1)
        Ky = (E*Iz/L**3) * np.array([[ 12,    6*L,  -12,    6*L],
                                     [6*L, 4*L**2, -6*L, 2*L**2],
                                     [-12,   -6*L,   12,   -6*L],
                                     [6*L, 2*L**2, -6*L, 4*L**2]])
        Kz = (E*Iy/L**3) * np.array([[  12,   -6*L,  -12,   -6*L],
                                     [-6*L, 4*L**2,  6*L, 2*L**2],
                                     [ -12,    6*L,   12,    6*L],
                                     [-6*L, 2*L**2,  6*L, 4*L**2]])

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



