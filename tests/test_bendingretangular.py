import numpy as np
from compmec.strct.material import Isotropic
from compmec.strct.section import Retangular
from compmec.strct.beam import EulerBernoulli
from compmec.strct.solver import solve
from usefulfunc import *
import pytest

@pytest.mark.dependency()
def test_begin():
    pass

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_bendingX():
    ntests = 10
    for i in range(ntests):
        E = random_between(1, 2)
        nu = random_between(0, 0.49)
        h = random_between(1, 2)
        b = random_between(1, 2)
        b = h = 1
        L = random_between(1, 2)
        Py = random_between(-1, 1)
        Pz = random_between(-1, 1)
        steel = Isotropic(E=E, nu=nu)
        rectangle = Retangular(b=b, h=h, nu=nu)
        A = (0, 0, 0)
        B = (L, 0, 0)
        bar = EulerBernoulli([A, B])
        bar.material = steel
        bar.section = rectangle
        bar.v = (0, 0, 1)

        U = np.empty((2, 6), dtype="object")
        F = np.zeros((2, 6))

        U[0, :] = 0
        F[1, 1] = Py
        F[1, 2] = Pz
        K = bar.stiffness_matrix()
        Utest, Ftest = solve(K, F, U)

        Iz = b*h**3/12
        Iy = b**3*h/12
        uy = Py*L**3/(3*E*Iz)
        uz = Pz*L**3/(3*E*Iy)
        ty = Pz*L**2/(2*E*Iy)
        tz = Py*L**2/(2*E*Iz)
        My = Pz*L
        Mz = Py*L
        Ugood = np.array([[0, 0, 0, 0, 0, 0],
                          [0, uy, uz, 0, -ty, tz]])
        Fgood = np.array([[0, -Py, -Pz, 0, My, -Mz],
                          [0, Py, Pz, 0, 0, 0]])
        np.testing.assert_almost_equal(Utest, Ugood)
        np.testing.assert_almost_equal(Ftest, Fgood)


@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_bendingY():
    ntests = 10
    for i in range(ntests):
        E = random_between(1, 2)
        nu = random_between(0, 0.49)
        h = random_between(1, 2)
        b = random_between(1, 2)
        b = h = 1
        L = random_between(1, 2)
        Px = random_between(-1, 1)
        Pz = random_between(-1, 1)
        steel = Isotropic(E=E, nu=nu)
        rectangle = Retangular(b=b, h=h, nu=nu)
        A = (0, 0, 0)
        B = (0, L, 0)
        bar = EulerBernoulli([A, B])
        bar.material = steel
        bar.section = rectangle
        bar.v = (0, 0, 1)

        U = np.empty((2, 6), dtype="object")
        F = np.zeros((2, 6))

        U[0, :] = 0
        F[1, 0] = Px
        F[1, 2] = Pz
        K = bar.stiffness_matrix()
        Utest, Ftest = solve(K, F, U)

        Iz = b*h**3/12
        Ix = b**3*h/12
        ux = Px*L**3/(3*E*Iz)
        uz = Pz*L**3/(3*E*Ix)
        tx = Pz*L**2/(2*E*Ix)
        tz = Px*L**2/(2*E*Iz)
        Mx = Pz*L
        Mz = Px*L
        Ugood = np.array([[0, 0, 0, 0, 0, 0],
                          [ux, 0, uz, tx, 0, -tz]])
        Fgood = np.array([[-Px, 0, -Pz, -Mx, 0, Mz],
                          [Px, 0, Pz, 0, 0, 0]])
        np.testing.assert_almost_equal(Utest, Ugood)
        np.testing.assert_almost_equal(Ftest, Fgood)

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_bendingY", "test_bendingY"])
def test_bendingXY():
    ntests = 10
    for i in range(ntests):
        E = random_between(1, 2)
        nu = random_between(0, 0.49)
        h = random_between(1, 2)
        b = random_between(1, 2)
        b = h = 1
        L = random_between(1, 2)
        Pw = random_between(-1, 1)
        Pv = random_between(-1, 1)
        r = random_unit_vector([True, True, False])
        v = np.array([0, 0, 1])
        w = np.cross(v, r)
        steel = Isotropic(E=E, nu=nu)
        rectangle = Retangular(b=b, h=h, nu=nu)
        A = (0, 0, 0)
        B = L * r
        bar = EulerBernoulli([A, B])
        bar.material = steel
        bar.section = rectangle
        bar.v = v

        U = np.empty((2, 6), dtype="object")
        F = np.zeros((2, 6))

        U[0, :] = 0
        F[1, :3] += Pw * w
        F[1, :3] += Pv * v
        K = bar.stiffness_matrix()
        Utest, Ftest = solve(K, F, U)

        Iz = b*h**3/12
        Ir = b**3*h/12
        uw = Pw*L**3/(3*E*Iz)
        uv = Pv*L**3/(3*E*Ir)
        tw = Pv*L**2/(2*E*Ir)
        tv = Pw*L**2/(2*E*Iz)
        Mw = Pv*L
        Mv = Pw*L
        Ugood = np.zeros((2, 6))
        Fgood = np.zeros((2, 6))
        Ugood[1, :3] += uw * w + uv * v
        Ugood[1, 3:] += -tw * w + tv * v
        Fgood[0, :3] -= Pw * w + Pv * v
        Fgood[0, 3:] += Mw * w - Mv * v
        Fgood[1, :3] += Pw * w + Pv * v
        np.testing.assert_almost_equal(Utest, Ugood)
        np.testing.assert_almost_equal(Ftest, Fgood)

if __name__ == "__main__":
    pytest.main()