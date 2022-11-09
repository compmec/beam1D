import numpy as np
from typing import Iterable, List, Tuple, Optional, Union
from compmec.nurbs import SplineBaseFunction, SplineCurve 
from compmec.nurbs.degreeoperations import degree_elevation_basefunction, degree_elevation_controlpoints
from compmec.nurbs.knotoperations import insert_knot_basefunction, insert_knot_controlpoints
import abc

class Point(object):
    __instances = []

    @staticmethod
    def validation_point(value: Tuple[float]):
        value = np.ndarray(value, dtype="float64")
        if value.ndim != 1:
            raise ValueError("A point must be a 1D-array")
        if len(value) != 3:
            raise ValueError("The point must be a 3D-point, with three values")

    def __new__(cls, value: Tuple[float]):
        if len(Point.__instances) == 0:
            return self.new(value)
        id = Point.get_id(value)
        if id is None:
            return self.new(value)
        return Point.__instances[id]

    @staticmethod
    def new(value: Tuple[float]):
        self = object.__new__(cls)
        Point.__instances.append(self)
        return self    

    @staticmethod
    def get_id(value: Tuple[float], distmax: float = 1e-9) -> int:
        """
        Precisa testar
        """
        if len(Point.__instances) == 0:
            return None
        value = np.array(value)
        distances = np.array([np.linalg.norm(point-value) for point in Point.__instances])
        mask = (distances < distmax)
        if not np.any(mask):
            return None
        return np.where(mask)[0][0]

    def __init__(self, value: Tuple[float]):
        self.__p = np.array(value, dtype="float64")
        self.__r = np.zeros(3, dtype="float64")
        self.__id = len(Point.__instances)

    @property
    def id(self):
        return self.__id

    @property
    def p(self):
        return self.__p

    @property
    def r(self):
        return self.__r
    
    def __str__(self):
        return str(self.p)
    
    def __repr__(self):
        return str(self)

    def __tuple__(self):
        return tuple(self.p)

    def __list__(self):
        return list(self.p)
    



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
    def __init__(self, path: Union[SplineCurve, np.ndarray]):
        self._dofs = None
        self.__init_path(path)

    def __init_path(self, path: Union[SplineCurve, np.ndarray]):
        if isinstance(path, SplineCurve):
            self.__curve = path
        else:
            P = np.array(path)
            if P.ndim != 2:
                raise ValueError(f"The received points must be an 2D array. Received {P.ndim}")
            npts, dim = P.shape
            if dim != 3:
                raise ValueError(f"The dimension of points must be 3. Received {dim}")
            degree = 1  # Degree of polynomial
            U = [0]*degree + list(np.linspace(0, 1, npts-degree+1))+ [1]*degree
            N = SplineBaseFunction(U)
            C = SplineCurve(N, P)
            self.__curve = C
        self.__ts = []
        for t in self.__curve.U:
            if t not in self.__ts:
                self.__ts.append(t)
        

    @property
    def path(self) -> SplineCurve:
        return self.__curve

    def normal(self, t:float ) -> np.ndarray:
        t = float(t)
        dpathdt = self.__curve.derivate()
        value = dpathdt(t)
        return value/np.linalg.norm(value)

    def evaluate(self, ts: Iterable[float], deformed: Optional[bool] = False) -> List[Tuple[float]]:
        results = []
        for t in ts:
            if deformed:
                if self.__deformedcurve is None:
                    displacement = self.field("u")
                    self.__deformedcurve = displacement
                    self.__deformedcurve.P += self.__curve.P  # Sum control points
                result = tuple(self.__deformedcurve(float(t)))
            else:
                result = tuple(self.__curve(float(t)))
            results.append(result)
        return results

    def addt(self, t: float):
        if t in self.__ts:
            return
        F = self.__curve.F
        P = self.__curve.P
        P = insert_knot_controlpoints(F, P, t)
        F = insert_knot_basefunction(F, t)
        self.__curve = self.__curve.__class__(F, P)
        self.__ts.append(t)
        self.__ts.sort()

    @property
    def dofs(self) -> int:
        return self._dofs
    
    @property
    def ts(self) -> np.ndarray:
        return np.array(self.__ts)

    @property
    def points(self) -> np.ndarray:
        return np.array([self.path(ti) for ti in self.__ts])

    @property
    def material(self) -> Material:
        return self.__material

    @property
    def section(self) -> Section:
        return self.__section

    @section.setter
    def section(self, value: Section):
        self.__section = value

    @material.setter
    def material(self, value: Material):
        self.__material = value

    def stiffness_matrix(self) -> np.ndarray:
        return self.global_stiffness_matrix()

    def field(self, fieldname: str):
        raise NotImplementedError("This function must be overwritten")



class ComputeField(object):

    def __validation_Structural1D(element: Structural1D):
        if not isinstance(element, Structural1D):
            raise TypeError(f"The given element must be a Structural1D instance. Received {type(element)}")