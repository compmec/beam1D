from typing import Tuple, Union

import numpy as np

from compmec.strct.__classes__ import Point


class PointBase(Point):
    @staticmethod
    def validation_creation(point: tuple[float]):
        if not isinstance(point, (list, tuple, np.ndarray)):
            raise TypeError("Point3D must be created from list/tuple/numpy array")
        for i, pi in enumerate(point):
            if not isinstance(pi, (int, float)):
                error_msg = f"To search the point, every coordinate must be a float.\n"
                error_msg += f"    type(point[{i}]) = {type(pi)}"
                raise TypeError(error_msg)

    def __add__(self, other: tuple[float]):
        return self.__class__([pi + qi for pi, qi in zip(self, other)])

    def __sub__(self, other: tuple[float]):
        return self.__class__([pi - qi for pi, qi in zip(self, other)])

    def __radd__(self, other: tuple[float]):
        return self.__class__([pi + qi for pi, qi in zip(self, other)])

    def __rsub__(self, other: tuple[float]):
        return self.__class__([qi - pi for pi, qi in zip(self, other)])

    def __eq__(self, other: tuple[float]):
        if isinstance(other, self.__class__):
            pass
        elif not isinstance(other, (tuple, list, np.ndarray)):
            return False
        try:
            other = self.__class__(other)
        except Exception as e:
            return False
        for pi, qi in zip(self, other):
            if pi != qi:
                return False
        return True

    def __ne__(self, other: tuple[float]):
        return not self.__eq__(other)


class Point2D(PointBase, Tuple):
    @staticmethod
    def validation_creation(point: tuple[float]):
        PointBase.validation_creation(point)
        if len(point) != 2:
            error_msg = "Point2D must be created with three float values. len(point) = {len(point)}"
            raise ValueError(error_msg)

    def __new__(cls, point: tuple[float]):
        if isinstance(point, Point2D):
            return point
        Point2D.validation_creation(point)
        return super(Point2D, cls).__new__(cls, tuple(point))


class Point3D(PointBase, Tuple):
    @staticmethod
    def validation_creation(point: tuple[float]):
        PointBase.validation_creation(point)
        if len(point) != 3:
            error_msg = "Point3D must be created with three float values. len(point) = {len(point)}"
            raise ValueError(error_msg)

    def __new__(cls, point: tuple[float]):
        if isinstance(point, Point3D):
            return point
        Point3D.validation_creation(point)
        return super(Point3D, cls).__new__(cls, tuple(point))


class Geometry1D(object):
    def __init__(self):
        self._all_points = []

    @property
    def points(self):
        return np.array(self._all_points)

    @property
    def npts(self):
        return len(self.points)

    def find_point(
        self, point: Tuple[float], tolerance: float = 1e-6
    ) -> Union[int, None]:
        """
        Given a point like (0.1, 3.1, 5), it returns the index of this point.
        If the point is too far (bigger than tolerance), it returns None
        """
        if not isinstance(tolerance, (int, float)):
            raise TypeError("Tolerance to find point must be a float")
        if tolerance <= 0:
            raise ValueError("Tolerance must be positive!")
        if self.npts == 0:
            return None
        point = Point3D(point)
        return self._find_point(point, tolerance)

    def _find_point(self, point: Point3D, tolerance: float) -> Union[int, None]:
        """
        Internal unprotected function. See docs of the original function
        """
        distances = [sum((pi - point) ** 2) for pi in self.points]
        distsquare = np.array(distances, dtype="float64")
        mindistsquare = np.min(distsquare)
        if np.all(mindistsquare > tolerance**2):
            return None
        indexs = np.where(distsquare == mindistsquare)[0]
        if len(indexs) > 1:
            raise ValueError("There's more than 1 point at the same position")
        return int(indexs[0])

    def create_point(self, point: Tuple[float]) -> int:
        """
        Creates a new point, and add it into geometry.
        Returns the index of the new created point.
        If the point exists, it gives ValueError
        """
        point = Point3D(point)
        index = self.find_point(point)
        if index is not None:
            raise ValueError(f"The received point already exists at index {index}")
        return self._create_point(point)

    def _create_point(self, point: Point3D) -> int:
        self._all_points.append(point)
        return self.npts - 1
