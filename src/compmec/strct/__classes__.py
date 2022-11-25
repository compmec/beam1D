import abc
from typing import Callable, Iterable, List, Optional, Tuple, Union

import compmec.nurbs as nurbs
import numpy as np


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
        distances = np.array(
            [np.sum((point.p - value) ** 2) for point in Point.__instances]
        )
        mask = distances < distmax
        if not np.any(mask):
            return None
        return np.where(mask)[0][0]

    def __init__(self, value: Tuple[float]):
        self.__p = np.array(value, dtype="float64")
        self.__r = np.zeros(3, dtype="float64")
        self.__id = len(Point.__instances) - 1

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


class Profile(abc.ABC):
    pass


class Material(object):
    def __init__(self):
        super().__init__(self)


class Section(abc.ABC):
    pass


class HomogeneousSection(Section):
    def __init__(self, material: Material, profile: Profile):
        self.material = material
        self.profile = profile
        self.__A = None
        self.__I = None

    @property
    def material(self) -> Material:
        return self.__material

    @property
    def profile(self) -> Profile:
        return self.__profile

    @property
    def A(self) -> Tuple[float, float, float]:
        if self.__A is None:
            self.compute_areas()
        return tuple(self.__A)

    @property
    def I(self) -> Tuple[float, float, float]:
        if self.__I is None:
            self.compute_inertias()
        return tuple(self.__I)

    @A.setter
    def A(self, value: Tuple[float, float, float]):
        self.__A = tuple(value)

    @I.setter
    def I(self, value: Tuple[float, float, float]):
        self.__I = tuple(value)

    @material.setter
    def material(self, value: Material):
        if not isinstance(value, Material):
            raise TypeError
        self.__material = value

    @profile.setter
    def profile(self, value: Profile):
        if not isinstance(value, Profile):
            raise TypeError
        self.__profile = value

    @abc.abstractmethod
    def compute_areas(self):
        raise NotImplementedError

    @abc.abstractmethod
    def compute_inertias(self):
        raise NotImplementedError


class Structural1DInterface(abc.ABC):
    @abc.abstractmethod
    def __init__(self, path: Union[nurbs.SplineCurve, np.ndarray]):
        raise NotImplementedError

    @property
    def path(self) -> nurbs.SplineCurve:
        return self.__path

    @property
    def ts(self) -> Tuple[float]:
        return tuple(set(self.path.U))

    @property
    def section(self) -> Section:
        return self.__section

    @property
    def field(self) -> Callable[[str], nurbs.SplineCurve]:
        """Returns function which receives a string and returns an nurbs.SplineCurve"""
        try:
            return self.__field
        except AttributeError as e:
            raise ValueError("You must run the simulation before calling 'field'")

    def set_path(self, value: nurbs.SplineCurve):
        if not isinstance(value, nurbs.SplineCurve):
            raise TypeError
        self.__path = value

    def set_section(self, value: Section):
        if not isinstance(value, Section):
            raise TypeError
        self.__section = value

    def set_field(self, value):
        self.__field = value


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
            raise ValueError(
                f"To set results: result.shape[0] = {value.shape[0]} != {npts} = npts"
            )
        if value.shape[1] != 6:
            raise ValueError(
                f"The number of results in must be {6}, received {value.shape[1]}"
            )
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
        """Computes the rotation of each point"""
        raise NotImplementedError

    @abc.abstractmethod
    def internalmomentum(self) -> nurbs.SplineCurve:
        """Computes the internal momentum of the beam"""
        raise NotImplementedError

    @abc.abstractmethod
    def externalmomentum(self) -> nurbs.SplineCurve:
        """Computes the external momentum applied on the beam"""
        raise NotImplementedError
