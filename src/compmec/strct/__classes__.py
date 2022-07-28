import numpy as np
from compmec.nurbs import SplineCurve, SplineBaseFunction

class Material(object):
    def __init__(self):
        super().__init__(self)

class Section(object):
    def __init__(self):
        self._A = np.zeros(3, dtype="float64")
        self._I = np.zeros(3, dtype="float64")

    @property
    def Ax(self) -> float:
        return self._A[0]

    @property
    def Ay(self) -> float:
        return self._A[1]

    @property
    def Az(self) -> float:
        return self._Az[2]

    @property
    def A(self) -> np.ndarray:
        return self._A
    
    
    @property
    def Ix(self) -> float:
        return self._I[0]

    @property
    def Iy(self) -> float:
        return self._I[1]

    @property
    def Iz(self) -> float:
        return self._I[2]

    @property
    def I(self) -> np.ndarray:
        return self._I


    @Ax.setter
    def Ax(self, value:float):
        self._A[0] = value

    @Ay.setter
    def Ay(self, value:float):
        self._A[1] = value

    @Az.setter
    def Az(self, value:float):
        self._A[2] = value

    @Ix.setter
    def Ix(self, value:float):
        self._I[0] = value

    @Iy.setter
    def Iy(self, value:float):
        self._I[1] = value

    @Iz.setter
    def Iz(self, value:float):
        self._I[2] = value

    def triangular_mesh(self, elementsize:float):
        raise NotImplementedError("This function must be redefined by child class")

    def mesh(self, elementsize:float = None):
        if elementsize is None:
            elementsize = 0.1*np.sqrt(self.Ax)
        return self.triangular_mesh(elementsize)


class Structural1D(object):
    def __init__(self, path):
        if isinstance(path, (tuple, list)):
            U = [0, 0, 1, 1]  # p = 1, n = 2
            N = SplineBaseFunction(U)
            P = np.zeros((2, 3))
            P[0, :len(path[0])] += path[0]
            P[1, :len(path[1])] += path[1]
            curve = SplineCurve(N, P)
            self.__curve = curve
        else:
            raise TypeError("Not expected received argument")
        self._ts = [0, 1]

    def path(self, t: float) -> np.ndarray:
        try:
            t = float(t)
        except Exception as e:
            raise TypeError(f"The parameter t must be a float. Could not convert {type(t)}")
        if t < 0 or t > 1:
            raise ValueError("t in path must be in [0, 1]")
        if t not in self._ts:
            self.addt(t)
        result = self.__curve(t)
        if result.ndim == 2:
            result = result.reshape(3)
        return result

    def defo(self, t: float) -> np.ndarray:
        try:
            t = float(t)
        except Exception as e:
            raise TypeError(f"The parameter t must be a float. Could not convert {type(t)}")
        if t < 0 or t > 1:
            raise ValueError("t in path must be in [0, 1]")
        result = self._defo(t)
        if result.ndim == 2:
            result = result.reshape(3)
        return result

    def addt(self, t: float):
        # self.__curve.knotinsert(t)
        self._ts.append(t)
        self._ts.sort()

    @property
    def dofs(self) -> int:
        return self.__dofs
    
    @property
    def ts(self) -> np.ndarray:
        return np.array(self._ts)

    @property
    def points(self) -> np.ndarray:
        return np.array([self.path(ti) for ti in self._ts])

    @property
    def material(self) -> Material:
        return self._material

    @property
    def section(self) -> Section:
        return self._section

    @section.setter
    def section(self, value: Section):
        self._section = value

    @material.setter
    def material(self, value: Material):
        self._material = value

    def stiffness_matrix(self) -> np.ndarray:
        return self.global_stiffness_matrix()

    def set_result(self, U: np.ndarray):
        if not isinstance(U, np.ndarray):
            raise TypeError("U must be a numpy array")
        if U.shape != (len(self.ts), self.dofs):
            raise ValueError(f"U shape must be ({len(self.ts)}, {self.dofs})")
        
        self._defo = SplineCurve()