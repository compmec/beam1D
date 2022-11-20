import numpy as np
from typing import Iterable, List, Tuple, Optional, Union, Callable
import compmec.nurbs as nurbs 
import abc

class Point(object):
    __instances = []
    __points = []

    @staticmethod
    def validation_point(value: Tuple[float]):
        value = np.array(value, dtype="float64")
        if value.ndim != 1:
            raise ValueError("A point must be a 1D-array")
        if len(value) != 3:
            raise ValueError("The point must be a 3D-point, with three values")

    def __new__(cls, value: Tuple[float]):
        if len(Point.__instances) == 0:
            return Point.new(value)
        id = Point.get_id(value)
        if id is None:
            return Point.new(value)
        return Point.__instances[id]

    @staticmethod
    def new(value: Tuple[float]):
        self = object.__new__(Point)
        Point.__instances.append(self)
        return self

    @staticmethod
    def get_id(value: Tuple[float], distmax: float = 1e-9) -> int:
        """
        Precisa testar
        """
        if len(Point.__instances) == 0:
            return None
        value = np.array(value, dtype="float64")
        distances = np.array([np.sum((point.p-value)**2) for point in Point.__instances])
        mask = (distances < distmax)
        if not np.any(mask):
            return None
        return np.where(mask)[0][0]

    def __init__(self, value: Tuple[float]):
        self.__p = np.array(value, dtype="float64")
        self.__r = np.zeros(3, dtype="float64")
        self.__id = len(Point.__instances)-1

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

    def __iter__(self):
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


class Structural1DInterface(abc.ABC):
    @abc.abstractmethod
    def __init__(self, path: Union[nurbs.SplineCurve, np.ndarray]):
        raise NotImplementedError

    @abc.abstractmethod
    def valid_path(self, value):
        raise NotImplementedError

    @abc.abstractmethod
    def valid_material(self, value):
        raise NotImplementedError

    @abc.abstractmethod
    def valid_section(self, value):
        raise NotImplementedError

    @abc.abstractmethod
    def valid_field(self, value):
        raise NotImplementedError

    @abc.abstractmethod
    def valid_t(self, value):
        raise NotImplementedError

    @property
    def path(self) -> nurbs.SplineCurve:
        return self.__path

    @property
    def ts(self) -> Tuple[float]:
        return tuple(self.__ts)

    @property
    def material(self) -> Material:
        return self.__material

    @property
    def section(self) -> Section:
        return self.__section

    @property
    def field(self) -> Callable[[str], nurbs.SplineCurve]:
        """Returns function which receives an string and returns an nurbs.SplineCurve"""
        try:
            return self.__field
        except AttributeError as e:
            raise ValueError("You must run the simulation before calling 'field'")


    @path.setter
    def path(self, value: nurbs.SplineCurve):
        self.valid_path(value)
        self.__path = value

    @material.setter
    def material(self, value: Material):
        self.valid_material(value)
        self.__material = value
    
    @section.setter
    def section(self, value: Section):
        self.valid_section(value)
        self.__section = value
        
    @field.setter
    def field(self, value):
        """After the simulation we use it to make an reference to the field computer"""
        self.valid_field(value)
        self.__field = value

    @abc.abstractmethod
    def addt(self, t: float):
        raise NotImplementedError

class TrussInterface(Structural1DInterface):
    @abc.abstractmethod
    def stiffness_matrix(self) -> np.ndarray:
        raise NotImplementedError

class BeamInterface(Structural1DInterface):
    @abc.abstractmethod
    def stiffness_matrix(self) -> np.ndarray:
        raise NotImplementedError
    



    

class ComputeFieldInterface(abc.ABC):
    
    
    @abc.abstractmethod
    def __init__(self, element: Structural1DInterface, result: np.ndarray):
        raise NotImplementedError
    
    @abc.abstractmethod
    def __call__(self, fieldname: str) -> nurbs.SplineCurve:
        raise NotImplementedError

    @abc.abstractmethod
    def field(self, fieldname: str) -> nurbs.SplineCurve:
        raise NotImplementedError
    
    @property
    def element(self) -> Structural1DInterface:
        return self.__element

    @property
    def result(self) -> np.ndarray:
        return np.copy(self.__result)

    @element.setter
    def element(self, value: Structural1DInterface):
        if not isinstance(value, Structural1DInterface):
            raise TypeError("The element must be a Structural1D instance")
        self.__element = value
    
    @result.setter
    def result(self, value: np.ndarray):
        if self.element is None:
            raise ValueError("To set result, you must set element first")
        ctrlpts = self.element.path.P
        npts, dim = ctrlpts.shape
        if value.shape[0] != npts:
            raise ValueError(f"To set results: result.shape[0] = {value.shape[0]} != {npts} = npts")
        if value.shape[1] != 6:
            raise ValueError(f"The number of results in must be {6}, received {value.shape[1]}")
        self.__result = value

    @abc.abstractmethod
    def position(self):
        """Compute the position of neutral line"""
        raise NotImplementedError

    @abc.abstractmethod
    def deformed(self):
        """Compute the deformed position of neutral line"""
        raise NotImplementedError

    @abc.abstractmethod
    def displacement(self) -> nurbs.SplineCurve:
        """Compute the displacement of each point"""
        raise NotImplementedError

    @abc.abstractmethod
    def externalforce(self) -> nurbs.SplineCurve:
        """Compute the external force applied on the element"""
        raise NotImplementedError

    @abc.abstractmethod
    def internalforce(self) -> nurbs.SplineCurve:
        """Compute the internal force inside the element"""
        raise NotImplementedError

    @abc.abstractmethod
    def vonmisesstress(self) -> nurbs.SplineCurve:
        """Compute the Von Mises Stress of the element"""
        raise NotImplementedError

    @abc.abstractmethod
    def trescastress(self) -> nurbs.SplineCurve:
        """Compute the Tresca Stress of the element"""
        raise NotImplementedError

class ComputeFieldTrussInterface(ComputeFieldInterface):
    pass

class ComputeFieldBeamInterface(ComputeFieldInterface):
    
    @abc.abstractmethod
    def rotations(self) -> nurbs.SplineCurve:
        """Computes the rotation of each point """
        raise NotImplementedError

    @abc.abstractmethod
    def internalmomentum(self) -> nurbs.SplineCurve:
        """Computes the internal momentum of the beam"""
        raise NotImplementedError

    @abc.abstractmethod
    def externalmomentum(self) -> nurbs.SplineCurve:
        """Computes the external momentum applied on the beam"""
        raise NotImplementedError