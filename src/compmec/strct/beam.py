
"""
Each point has 6 unknowns:

"""
import numpy as np
from compmec.strct.material import Material
from compmec.strct.section import Section

class Structural1D(object):
    def __init__(self, path):
        if isinstance(path, (tuple, list)):
            self._p0 = np.array(path[0])
            self._p1 = np.array(path[1])
        elif callable(path):
            self._p0 = np.array(path(0))
            self._p1 = np.array(path(1))
        else:
            raise TypeError("Not expected received argument")
        self._p = self._p1 - self._p0
        self._L = np.sqrt(np.sum(self._p**2))

    @property
    def p(self) -> np.ndarray:
        return self._p

    @property
    def L(self) -> float:
        return self._L

    @property
    def material(self) -> Material:
        return self._material


    @material.setter
    def material(self, value:Material):
        self._material = value

    @property
    def section(self) -> Section:
        return self._section

    @section.setter
    def section(self, value : Section):
        self._section = value

    def rotation_matrix33(self):
        px, py, pz = self.p
        cos = px/self.L
        pyz = py**2 + pz**2
        if cos == 1:
            return np.eye(3)
        elif cos == -1:
            return -np.eye(3)
        R33 = np.array([[0, 0, 0],
                        [0, pz**2, -py*pz],
                        [0, -py*pz, py**2]], dtype="float64")
        R33 *= (1-cos)/pyz
        R33 += np.array([[px, py, pz],
                         [-py, px, 0],
                         [-pz, 0, px]])/self.L
        return R33

class Truss(Structural1D):
    def __init__(self, path):
        super().__init__(path)


class Cable(Structural1D):
    def __init__(self, path):
        super().__init__(path)

class Beam(Structural1D):
    def __init__(self, path):
        """
        vertical is the perpendicular direction to the line
        ver
        """
        super().__init__(path)
        

    @property
    def v(self) -> np.ndarray:
        if self._v is None:
            raise Exception("Cannot give v because it's None")
        return self._v

    @v.setter
    def v(self, value: np.ndarray) -> None:
        if np.sum(value * self.r) != 0:
            raise Exception("vertical must perpendicular to the beams line")
        self._v = value
        self._v /= np.sqrt(np.sum(self._v**2))

    def set_random_v(self):
        self._v = np.random.rand(3)
        self._v -= np.sum(self._v*self.p) * self.p
        self._v /= np.sqrt(np.sum(self._v**2))
    

    def global_stiffness_matrix(self):
        Kloc = self.local_stiffness_matrix()
        R33 = self.rotation_matrix33()
        Rexp = np.zeros((12, 12), dtype="float64")
        Rexp[:3, :3] = R33[:, :].T
        Rexp[3:6, 3:6] = R33[:, :].T
        Rexp[6:9, 6:9] = R33[:, :].T
        Rexp[9:, 9:] = R33[:, :].T
        Klocexp = Kloc.reshape((12, 12))
        Kgloexp = Rexp @ Klocexp @ Rexp.T
        Kglo = Kgloexp.reshape((2, 6, 2, 6))
        return Kglo

class EulerBernoulli(Beam):
    def __init__(self, path):
        super().__init__(path)

    def stiffness_matrix(self):
        return self.global_stiffness_matrix()

    def local_stiffness_matrix(self):
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
        for i in range(2):
            for j in range(2):
                K[i, 0, j, 0] = Kx[i, j]
        for i in range(2):
            for j in range(2):
                K[i, 3, j, 3] = Kt[i, j]
        for i in range(2):
            for j in range(2):
                for wa, a in enumerate([1, 5]):
                    for wb, b in enumerate([1, 5]):
                        K[i, a, j, b] = Ky[2*i+wa, 2*j+wb]
        for i in range(2):
            for j in range(2):
                for wa, a in enumerate([2, 4]):
                    for wb, b in enumerate([2, 4]):
                        K[i, a, j, b] = Kz[2*i+wa, 2*j+wb]
        # K = K.reshape((12, 12))
        return K


class Timoshenko(Beam):
    def __init__(self, path):
        super().__init__(path)



