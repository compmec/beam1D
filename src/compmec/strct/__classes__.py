import numpy as np
from geomdl import BSpline

class Material(object):
    def __init__(self):
        super().__init__(self)

class Section(object):
    def __init__(self, nu: float):
        self.nu = nu
        self.__A = None
        self.__I = None

    def init_zeros_A(self):
        self.__A = np.zeros(3, dtype="float64")

    def init_zeros_I(self):
        self.__I = np.zeros(3, dtype="float64")

    @property
    def nu(self):
        return self.__nu

    @property
    def Ax(self) -> float:
        if self.__A is None:
            self.compute_areas()
        return self.__A[0]

    @property
    def Ay(self) -> float:
        if self.__A is None:
            self.compute_areas()
        return self.__A[1]

    @property
    def Az(self) -> float:
        if self.__A is None:
            self.compute_areas()
        return self.__A[2]

    @property
    def A(self) -> np.ndarray:
        if self.__A is None:
            self.compute_areas()
        return self.__A
    
    @property
    def Ix(self) -> float:
        if self.__I is None:
            self.compute_inertias()
        return self.__I[0]

    @property
    def Iy(self) -> float:
        if self.__I is None:
            self.compute_inertias()
        return self.__I[1]

    @property
    def Iz(self) -> float:
        if self.__I is None:
            self.compute_inertias()
        return self.__I[2]

    @property
    def I(self) -> np.ndarray:
        if self.__I is None:
            self.compute_inertias()
        return self.__I

    @nu.setter
    def nu(self, value: float):
        value = float(value)
        if value < 0:
            raise ValueError(f"The poisson value cannot be less than 0: nu={value}")
        if value > 0.5:
            raise ValueError(f"The poisson value cannot be greater than 0.5: nu={value}")
        self.__nu = value

    @Ax.setter
    def Ax(self, value: float):
        value = float(value)
        if value <= 0:
            raise ValueError(f"Cannot set a area as zero or negative: {value}")
        if self.__A is None:
            self.init_zeros_A()
        self.__A[0] = value

    @Ay.setter
    def Ay(self, value: float):
        if value <= 0:
            raise ValueError(f"Cannot set a area as zero or negative: {value}")
        if self.__A is None:
            self.init_zeros_A()
        self.__A[1] = value

    @Az.setter
    def Az(self, value: float):
        if value <= 0:
            raise ValueError(f"Cannot set a area as zero or negative: {value}")
        if self.__A is None:
            self.init_zeros_A()
        self.__A[2] = value

    @Ix.setter
    def Ix(self, value: float):
        if value <= 0:
            raise ValueError(f"Cannot set a inertia as zero or negative: {value}")
        if self.__I is None:
            self.init_zeros_I()
        self.__I[0] = value

    @Iy.setter
    def Iy(self, value: float):
        if value <= 0:
            raise ValueError(f"Cannot set a inertia as zero or negative: {value}")
        if self.__I is None:
            self.init_zeros_I()
        self.__I[1] = value

    @Iz.setter
    def Iz(self, value: float):
        if value <= 0:
            raise ValueError(f"Cannot set a inertia as zero or negative: {value}")
        if self.__I is None:
            self.init_zeros_I()
        self.__I[2] = value

    def shear_coefficient(self):
        raise NotImplementedError("This function must be overwritten")

    def compute_areas(self):
        raise NotImplementedError("This function must be overwritten")

    def compute_inertias(self):
        raise NotImplementedError("This function must be overwritten")

    def triangular_mesh(self, elementsize:float):
        raise NotImplementedError("This function must be redefined by child class")

    def mesh(self, elementsize:float = None):
        if elementsize is None:
            elementsize = 0.1*np.sqrt(self.Ax)
        return self.triangular_mesh(elementsize)


class Structural1D(object):
    def __init__(self, path):
        if isinstance(path, (tuple, list)):
            P = [[0, 0, 0],
                 [0, 0, 0]]
            for i, p in enumerate(path):
                for j, v in enumerate(p):
                    P[i][j] = v
            curve = BSpline.Curve()
            curve.degree = 1
            curve.ctrlpts = P
            curve.knotvector = [0, 0, 1, 1]
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
        result = self.__curve.evaluate_single(t)
        return tuple(result)

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
        