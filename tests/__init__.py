import numpy as np
import pytest


def random_vector():
    return np.random.rand(3)


def random_unit_vector():
    r = random_vector()
    return r / np.sqrt(np.sum(r * r))


def R(r):
    r = np.array(r)
    r /= np.linalg.norm(r)
    rx, ry, rz = r
    if rx == 1:
        return np.eye(3)
    if rx == -1:
        return -np.eye(3)
    cos = rx
    ryz = ry**2 + rz**2
    Rtot = np.array([[rx, ry, rz], [-ry, rx, 0], [-rz, 0, rx]], dtype="float64")
    Rtot += (
        np.array([[0, 0, 0], [0, rz**2, -ry * rz], [0, -ry * rz, ry**2]])
        * (1 - rx)
        / ryz
    )
    return Rtot


def test_randomUnitVector():
    for i in range(10):
        r = random_unit_vector()
        assert np.abs(np.sum(r**2) - 1) < 1e-9


def test_detR():
    for i in range(10):
        r = random_unit_vector()
        Rtot = R(r)
        det = np.linalg.det(Rtot)
        assert np.abs(det - 1) < 1e-9


def test_invR():
    for i in range(10):
        r = random_unit_vector()
        Rtot = R(r)
        Rinv = np.linalg.inv(Rtot)
        Mat = Rinv @ Rtot
        diff = Mat - np.eye(3)
        assert np.max(diff) < 1e-9


def test_Rtranspose():
    for i in range(10):
        r = random_unit_vector()
        Rtot = R(r)
        Rtra = Rtot.T
        II = Rtra @ Rtot
        diff = np.abs(II - np.eye(3))
        assert np.max(diff) < 1e-9


def test_Rapplytorknown():
    for rx in (-1, 0, 1):
        for ry in (-1, 0, 1):
            for rz in (-1, 0, 1):
                abso = np.sqrt(rx**2 + ry**2 + rz**2)
                if abso == 0:
                    continue
                r = np.array([rx, ry, rz]) / abso
                print("norm(r) = ", np.linalg.norm(r))
                Rtot = R(r)
                rt = Rtot @ r
                print("r = ", r)
                print("rt = ", rt)

                diff = rt - np.array([1, 0, 0])
                diff = np.abs(diff)
                assert np.max(diff) < 1e-9


def test_Rapplytor():
    for i in range(10):
        r = random_unit_vector()
        Rtot = R(r)
        rt = Rtot @ r
        diff = rt - np.array([1, 0, 0])
        diff = np.abs(diff)
        assert np.max(diff) < 1e-9


if __name__ == "__main__":
    pytest.main()
