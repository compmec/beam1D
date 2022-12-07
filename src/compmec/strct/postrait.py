from typing import Iterable, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt

from compmec.strct.__classes__ import Element1D
from compmec.strct.system import StaticSystem


def expand_ts(ts: Iterable[float]):
    expts = []
    for i in range(len(ts) - 1):
        newts = np.linspace(ts[i], ts[i + 1], 10, endpoint=False)
        expts.extend(list(newts))
    expts.append(ts[-1])
    return expts


def splinecurve_element(
    p: int,
    ts: Iterable[float],
    points: np.ndarray,
    solution: Optional[np.ndarray] = None,
):
    """
    p is the polynomial order we want
    ts is the vector of points of a beam [0.0, 0.1, ..., 0.9, 1.0]
    len(ts) = npts
    points is an array of shape [npts, 3]
    solution is an array of shape [npts, 6]
    """
    pts = np.copy(points)
    if solution is None:
        return curve_spline(p=p, u=[ts], points=[pts])
    der = np.zeros(points.shape)
    pts += solution[:, :3]
    der += solution[:, 3:]
    return curve_spline(p=p, u=[ts, ts], points=[pts, der])


def elementsolution(system: StaticSystem, element: Element1D):
    points = element.points
    indexs = np.zeros(points.shape[0], dtype="int32")
    for i, p in enumerate(points):
        indexs[i] = system._geometry.find_point(p)
    syssol = system.solution
    solution = syssol[indexs, :]
    return solution


if __name__ == "__main__":
    raise Exception("Cannot run this file. It should be imported")
