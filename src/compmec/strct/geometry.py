from typing import Tuple, Union

import numpy as np

from compmec.strct.__classes__ import Point


class Point3D(Point, Tuple):
    @staticmethod
    def validation_creation(point: tuple[float]):
        if not isinstance(point, (list, tuple, np.ndarray)):
            raise TypeError("Point3D must be created from list/tuple/numpy array")
        if len(point) != 3:
            error_msg = "Point3D must be created with three float values. len(point) = {len(point)}"
            raise ValueError(error_msg)
        for i, pi in enumerate(point):
            if not isinstance(pi, (int, float)):
                error_msg = f"To search the point, every coordinate must be a float.\n"
                error_msg += f"    type(point[{i}]) = {type(pi)}"
                raise TypeError(error_msg)

    def __new__(cls, point: tuple[float]):
        Point3D.validation_creation(point)
        return super(Point3D, cls).__new__(cls, tuple(point))

    def __add__(self, other: tuple[float]):
        return tuple([pi + qi for pi, qi in zip(self, other)])

    def __sub__(self, other: tuple[float]):
        return tuple([pi - qi for pi, qi in zip(self, other)])

    def __radd__(self, other: tuple[float]):
        return tuple([pi + qi for pi, qi in zip(self, other)])

    def __rsub__(self, other: tuple[float]):
        return tuple([qi - pi for pi, qi in zip(self, other)])


class Geometry1D(object):
    def __init__(self):
        self._all_points = []

    @property
    def dim(self):
        return 3

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
        if not isinstance(tolerance, float):
            raise TypeError("Tolerance to find point must be a float")
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
        print("dist square = ")
        print(distsquare)
        mindistsquare = np.min(distsquare)
        print(mindistsquare)
        if np.all(mindistsquare > tolerance):
            return None
        index = np.where(distsquare == mindistsquare)
        if len(index) > 1:
            raise ValueError("There's more than 1 point at the same position")
        return int(index[0])

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
