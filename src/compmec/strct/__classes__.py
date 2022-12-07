import abc
from typing import Callable, Iterable, List, Optional, Tuple, Union

import compmec.nurbs as nurbs
import numpy as np


class Point(object):
    _instances = []
    _points = []

    @staticmethod
    def validation_point(value: Tuple[float]):
        value = np.array(value, dtype="float64")
        if value.ndim != 1:
            raise ValueError("A point must be a 1D-array")
        if len(value) != 3:
            raise ValueError("The point must be a 3D-point, with three values")

    def _new_(cls, value: Tuple[float]):
        if len(Point._instances) == 0:
            return Point.new(value)
        id = Point.get_id(value)
        if id is None:
            return Point.new(value)
        return Point._instances[id]

    @staticmethod
    def new(value: Tuple[float]):
        self = object._new_(Point)
        Point._instances.append(self)
        return self

    @staticmethod
    def get_id(value: Tuple[float], distmax: float = 1e-9) -> int:
        """
        Precisa testar
        """
        if len(Point._instances) == 0:
            return None
        value = np.array(value, dtype="float64")
        distances = np.array(
            [np.sum((point.p - value) ** 2) for point in Point._instances]
        )
        mask = distances < distmax
        if not np.any(mask):
            return None
        return np.where(mask)[0][0]

    def __init__(self, value: Tuple[float]):
        self._p = np.array(value, dtype="float64")
        self._r = np.zeros(3, dtype="float64")
        self._id = len(Point._instances) - 1

    @property
    def id(self):
        return self._id

    @property
    def p(self):
        return self._p

    @property
    def r(self):
        return self._r

    def _str_(self):
        return str(self.p)

    def _repr_(self):
        return str(self)

    def _iter_(self):
        return tuple(self.p)

    def _list_(self):
        return list(self.p)


class Profile(abc.ABC):
    @abc.abstractmethod
    def area(self) -> float:
        raise NotImplementedError


class Material(object):
    def __init__(self):
        super().__init__(self)


class Section(abc.ABC):
    pass


class Element1D(abc.ABC):
    pass


class ComputeFieldInterface(abc.ABC):
    @abc.abstractmethod
    def __init__(self, element: Element1D, result: np.ndarray):
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, fieldname: str) -> nurbs.SplineCurve:
        raise NotImplementedError

    @abc.abstractmethod
    def field(self, fieldname: str) -> nurbs.SplineCurve:
        raise NotImplementedError

    @property
    def element(self) -> Element1D:
        return self._element

    @property
    def result(self) -> np.ndarray:
        return np.copy(self._result)

    @element.setter
    def element(self, value: Element1D):
        if not isinstance(value, Element1D):
            raise TypeError("The element must be a Element1D instance")
        self._element = value

    @result.setter
    def result(self, value: np.ndarray):
        if self.element is None:
            raise ValueError("To set result, you must set element first")
        ctrlpts = self.element.path.ctrlpoints
        npts, dim = ctrlpts.shape
        if value.shape[0] != npts:
            raise ValueError(
                f"To set results: result.shape[0] = {value.shape[0]} != {npts} = npts"
            )
        if value.shape[1] != 6:
            raise ValueError(
                f"The number of results in must be {6}, received {value.shape[1]}"
            )
        self._result = value

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
