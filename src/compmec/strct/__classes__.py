import abc
from typing import Callable, Iterable, List, Optional, Tuple, Union

import compmec.nurbs as nurbs
import numpy as np


class Point(object):
    pass


class Profile(abc.ABC):
    @abc.abstractmethod
    def area(self) -> float:
        raise NotImplementedError


class Material(object):
    pass


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


class Conector(object):
    pass
