import numpy as np
from compmec.strct.material import Isotropic
from compmec.strct.section import Circle
from compmec.strct.element import EulerBernoulli
from compmec.strct.solver import solve
import pytest
from usefulfunc import *


@pytest.mark.order(2)
@pytest.mark.dependency(
	depends=["tests/test_solver.py::test_end",
             "tests/test_material.py::test_end",
             "tests/test_structural1D.py::test_end",
             "tests/test_section.py::test_circle"],
    scope='session')
def test_begin():
    pass

@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_bendingXtoY():
    ntests = 10
    for i in range(ntests):
        E = random_between(1, 2)
        nu = random_between(0, 0.49)
        d = random_between(1, 2)
        L = random_between(1, 2)
        P = random_between(-1, 1)

        steel = Isotropic(E=E, nu=nu)
        circle = Circle(R=d/2, nu=nu)
        A = (0, 0, 0)
        B = (L, 0, 0)
        bar = EulerBernoulli([A, B])
        bar.material = steel
        bar.section = circle

        U = np.empty((2, 6), dtype="object")
        F = np.zeros((2, 6))

        U[0, :] = 0
        F[1, 1] = P
        K = bar.stiffness_matrix()
        Utest, Ftest = solve(K, F, U)


        uy = (64*P*L**3)/(3*np.pi*E*d**4)
        tz = 32*P*L**2/(np.pi*E*d**4)
        Mz = P*L
        Ugood = np.array([[0, 0, 0, 0, 0, 0],
                          [0, uy, 0, 0, 0, tz]])
        Fgood = np.array([[0, -P, 0, 0, 0, -Mz],
                          [0, P, 0, 0, 0, 0]])
        np.testing.assert_almost_equal(Utest, Ugood)
        np.testing.assert_almost_equal(Ftest, Fgood)


@pytest.mark.order(2)   
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin", "test_bendingXtoY"])
def test_bendingXtoZ():
    ntests = 10
    for i in range(ntests):
        E = random_between(1, 2)
        nu = random_between(0, 0.49)
        d = random_between(1, 2)
        L = random_between(1, 2)
        P = random_between(-1, 1)

        steel = Isotropic(E=E, nu=nu)
        circle = Circle(R=d/2, nu=nu)
        A = (0, 0, 0)
        B = (L, 0, 0)
        bar = EulerBernoulli([A, B])
        bar.material = steel
        bar.section = circle

        U = np.empty((2, 6), dtype="object")
        F = np.zeros((2, 6))
        U[0, :] = 0
        F[1, 2] = P

        K = bar.stiffness_matrix()
        Utest, Ftest = solve(K, F, U)

        uz = (64*P*L**3)/(3*np.pi*E*d**4)
        ty = 32*P*L**2/(np.pi*E*d**4)
        My = P*L
        Ugood = np.array([[0, 0, 0, 0, 0, 0],
                          [0, 0, uz, 0, -ty, 0]])
        Fgood = np.array([[0, 0, -P, 0, My, 0],
                          [0, 0, P, 0, 0, 0]])
        np.testing.assert_almost_equal(Utest, Ugood)
        np.testing.assert_almost_equal(Ftest, Fgood)


@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_bendingXtoY", "test_bendingXtoZ"])
def test_bendingXtoYZ():
    ntests = 10
    for i in range(ntests):
        E = random_between(1, 2)
        nu = random_between(0, 0.49)
        d = random_between(1, 2)
        L = random_between(1, 2)
        P = random_between(-1, 1)
        r = random_unit_vector([False, True, True])
        rx = np.cross(r, (1, 0, 0))

        steel = Isotropic(E=E, nu=nu)
        circle = Circle(R=d/2, nu=nu)
        A = (0, 0, 0)
        B = (L, 0, 0)
        bar = EulerBernoulli([A, B])
        bar.material = steel
        bar.section = circle

        U = np.empty((2, 6), dtype="object")
        F = np.zeros((2, 6))
        U[0, :] = 0
        F[1, :3] = P * r

        K = bar.stiffness_matrix()
        Utest, Ftest = solve(K, F, U)

        ur = (64*P*L**3)/(3*np.pi*E*d**4)
        tr = 32*P*L**2/(np.pi*E*d**4)
        Mr = P*L
        Ugood = np.zeros((2, 6))
        Fgood = np.zeros((2, 6))
        Ugood[1, :3] = ur * r
        Ugood[1, 3:] = -tr * rx
        Fgood[0, :3] = -P * r
        Fgood[0, 3:] = Mr * rx
        Fgood[1, :3] = P * r
        
        np.testing.assert_almost_equal(Utest, Ugood)
        np.testing.assert_almost_equal(Ftest, Fgood)


@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin", "test_bendingXtoY"])
def test_bendingYtoX():
    ntests = 10
    for i in range(ntests):
        E = random_between(1, 2)
        nu = random_between(0, 0.49)
        d = random_between(1, 2)
        L = random_between(1, 2)
        P = random_between(-1, 1)

        steel = Isotropic(E=E, nu=nu)
        circle = Circle(R=d/2, nu=nu)
        A = (0, 0, 0)
        B = (0, L, 0)
        bar = EulerBernoulli([A, B])
        bar.material = steel
        bar.section = circle

        U = np.empty((2, 6), dtype="object")
        F = np.zeros((2, 6))
        U[0, :] = 0
        F[1, 0] = P

        K = bar.stiffness_matrix()
        Utest, Ftest = solve(K, F, U)

        ux = (64*P*L**3)/(3*np.pi*E*d**4)
        tz = 32*P*L**2/(np.pi*E*d**4)
        Mz = P*L
        Ugood = np.array([[0, 0, 0, 0, 0, 0],
                          [ux, 0, 0, 0, 0, -tz]])
        Fgood = np.array([[-P, 0, 0, 0, 0, Mz],
                          [P, 0, 0, 0, 0, 0]])
        np.testing.assert_almost_equal(Utest, Ugood)
        np.testing.assert_almost_equal(Ftest, Fgood)

@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin", "test_bendingXtoY"])
def test_bendingYtoZ():
    ntests = 10
    for i in range(ntests):
        E = random_between(1, 2)
        nu = random_between(0, 0.49)
        d = random_between(1, 2)
        L = random_between(1, 2)
        P = random_between(-1, 1)

        steel = Isotropic(E=E, nu=nu)
        circle = Circle(R=d/2, nu=nu)
        A = (0, 0, 0)
        B = (0, L, 0)
        bar = EulerBernoulli([A, B])
        bar.material = steel
        bar.section = circle

        U = np.empty((2, 6), dtype="object")
        F = np.zeros((2, 6))
        U[0, :] = 0
        F[1, 2] = P

        K = bar.stiffness_matrix()
        Utest, Ftest = solve(K, F, U)

        uz = (64*P*L**3)/(3*np.pi*E*d**4)
        tx = 32*P*L**2/(np.pi*E*d**4)
        Mx = P*L
        Ugood = np.array([[0, 0, 0, 0, 0, 0],
                          [0, 0, uz, tx, 0, 0]])
        Fgood = np.array([[0, 0, -P, -Mx, 0, 0],
                          [0, 0, P, 0, 0, 0]])
        np.testing.assert_almost_equal(Utest, Ugood)
        np.testing.assert_almost_equal(Ftest, Fgood)


@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_bendingYtoX", "test_bendingYtoZ"])
def test_bendingYtoXZ():
    ntests = 10
    for i in range(ntests):
        E = random_between(1, 2)
        nu = random_between(0, 0.49)
        d = random_between(1, 2)
        L = random_between(1, 2)
        P = random_between(-1, 1)
        r = random_unit_vector([True, False, True])
        rx = np.cross(r, (0, 1, 0))

        steel = Isotropic(E=E, nu=nu)
        circle = Circle(R=d/2, nu=nu)
        A = (0, 0, 0)
        B = (0, L, 0)
        bar = EulerBernoulli([A, B])
        bar.material = steel
        bar.section = circle

        U = np.empty((2, 6), dtype="object")
        F = np.zeros((2, 6))
        U[0, :] = 0
        F[1, :3] = P * r

        K = bar.stiffness_matrix()
        Utest, Ftest = solve(K, F, U)

        ur = (64*P*L**3)/(3*np.pi*E*d**4)
        tr = 32*P*L**2/(np.pi*E*d**4)
        Mr = P*L
        Ugood = np.zeros((2, 6))
        Fgood = np.zeros((2, 6))
        Ugood[1, :3] = ur * r
        Ugood[1, 3:] = -tr * rx
        Fgood[0, :3] = -P * r
        Fgood[0, 3:] = Mr * rx
        Fgood[1, :3] = P * r
        
        np.testing.assert_almost_equal(Utest, Ugood)
        np.testing.assert_almost_equal(Ftest, Fgood)


@pytest.mark.order(2)      
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin", "test_bendingXtoY"])
def test_bendingZtoX():
    ntests = 10
    for i in range(ntests):
        E = random_between(1, 2)
        nu = random_between(0, 0.49)
        d = random_between(1, 2)
        L = random_between(1, 2)
        P = random_between(-1, 1)

        steel = Isotropic(E=E, nu=nu)
        circle = Circle(R=d/2, nu=nu)
        A = (0, 0, 0)
        B = (0, 0, L)
        bar = EulerBernoulli([A, B])
        bar.material = steel
        bar.section = circle

        U = np.empty((2, 6), dtype="object")
        F = np.zeros((2, 6))
        U[0, :] = 0
        F[1, 0] = P

        K = bar.stiffness_matrix()
        Utest, Ftest = solve(K, F, U)

        ux = (64*P*L**3)/(3*np.pi*E*d**4)
        ty = 32*P*L**2/(np.pi*E*d**4)
        My = P*L
        Ugood = np.array([[0, 0, 0, 0, 0, 0],
                          [ux, 0, 0, 0, ty, 0]])
        Fgood = np.array([[-P, 0, 0, 0, -My, 0],
                          [P, 0, 0, 0, 0, 0]])
        np.testing.assert_almost_equal(Utest, Ugood)
        np.testing.assert_almost_equal(Ftest, Fgood)


@pytest.mark.order(2)  
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin", "test_bendingXtoY"])
def test_bendingZtoY():
    ntests = 10
    for i in range(ntests):
        E = random_between(1, 2)
        nu = random_between(0, 0.49)
        d = random_between(1, 2)
        L = random_between(1, 2)
        P = random_between(-1, 1)

        steel = Isotropic(E=E, nu=nu)
        circle = Circle(R=d/2, nu=nu)
        A = (0, 0, 0)
        B = (0, 0, L)
        bar = EulerBernoulli([A, B])
        bar.material = steel
        bar.section = circle

        U = np.empty((2, 6), dtype="object")
        F = np.zeros((2, 6))
        U[0, :] = 0
        F[1, 1] = P

        K = bar.stiffness_matrix()
        Utest, Ftest = solve(K, F, U)

        uy = (64*P*L**3)/(3*np.pi*E*d**4)
        tx = 32*P*L**2/(np.pi*E*d**4)
        Mx = P*L
        Ugood = np.array([[0, 0, 0, 0, 0, 0],
                          [0, uy, 0, -tx, 0, 0]])
        Fgood = np.array([[0, -P, 0, Mx, 0, 0],
                          [0, P, 0, 0, 0, 0]])
        np.testing.assert_almost_equal(Utest, Ugood)
        np.testing.assert_almost_equal(Ftest, Fgood)


@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_bendingZtoX", "test_bendingZtoY"])
def test_bendingZtoXY():
    ntests = 10
    for i in range(ntests):
        E = random_between(1, 2)
        nu = random_between(0, 0.49)
        d = random_between(1, 2)
        L = random_between(1, 2)
        P = random_between(-1, 1)
        r = random_unit_vector([True, True, False])
        rx = np.cross(r, (0, 0, 1))

        steel = Isotropic(E=E, nu=nu)
        circle = Circle(R=d/2, nu=nu)
        A = (0, 0, 0)
        B = (0, 0, L)
        bar = EulerBernoulli([A, B])
        bar.material = steel
        bar.section = circle

        U = np.empty((2, 6), dtype="object")
        F = np.zeros((2, 6))

        U[0, :] = 0
        F[1, :3] = P * r
        K = bar.stiffness_matrix()
        Utest, Ftest = solve(K, F, U)

        ur = (64*P*L**3)/(3*np.pi*E*d**4)
        tr = 32*P*L**2/(np.pi*E*d**4)
        Mr = P*L
        Ugood = np.zeros((2, 6))
        Fgood = np.zeros((2, 6))
        Ugood[1, :3] = ur * r
        Ugood[1, 3:] = -tr * rx
        Fgood[0, :3] = -P * r
        Fgood[0, 3:] = Mr * rx
        Fgood[1, :3] = P * r
        
        np.testing.assert_almost_equal(Utest, Ugood)
        np.testing.assert_almost_equal(Ftest, Fgood)


@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_bendingXtoY", "test_bendingYtoX"])
def test_bendingXYtoXY():
    ntests = 10
    for i in range(ntests):
        E = random_between(1, 2)
        nu = random_between(0, 0.49)
        d = random_between(1, 2)
        L = random_between(1, 2)
        P = random_between(-1, 1)
        r = random_unit_vector([True, True, False])
        v = np.cross(r, (0, 0, 1))

        steel = Isotropic(E=E, nu=nu)
        circle = Circle(R=d/2, nu=nu)
        A = (0, 0, 0)
        B = L * r
        bar = EulerBernoulli([A, B])
        bar.material = steel
        bar.section = circle

        U = np.empty((2, 6), dtype="object")
        F = np.zeros((2, 6))

        U[0, :] = 0
        F[1, :3] = P * v
        K = bar.stiffness_matrix()
        Utest, Ftest = solve(K, F, U)

        uv = (64*P*L**3)/(3*np.pi*E*d**4)
        tz = 32*P*L**2/(np.pi*E*d**4)
        Mz = P*L
        Ugood = np.zeros((2, 6))
        Fgood = np.zeros((2, 6))
        Ugood[1, :3] = uv * v
        Ugood[1, 5] = -tz
        Fgood[0, :3] = -P * v
        Fgood[0, 5] = Mz
        Fgood[1, :3] = P * v
        np.testing.assert_almost_equal(Utest, Ugood)
        np.testing.assert_almost_equal(Ftest, Fgood)

@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_bendingXtoZ", "test_bendingZtoX"])
def test_bendingXZtoXZ():
    ntests = 10
    for i in range(ntests):
        E = random_between(1, 2)
        nu = random_between(0, 0.49)
        d = random_between(1, 2)
        L = random_between(1, 2)
        P = random_between(-1, 1)
        r = random_unit_vector([True, False, True])
        v = np.cross(r, (0, 1, 0))

        steel = Isotropic(E=E, nu=nu)
        circle = Circle(R=d/2, nu=nu)
        A = (0, 0, 0)
        B = L * r
        bar = EulerBernoulli([A, B])
        bar.material = steel
        bar.section = circle

        U = np.empty((2, 6), dtype="object")
        F = np.zeros((2, 6))
        U[0, :] = 0
        F[1, :3] = P * v

        K = bar.stiffness_matrix()
        Utest, Ftest = solve(K, F, U)

        uv = (64*P*L**3)/(3*np.pi*E*d**4)
        ty = 32*P*L**2/(np.pi*E*d**4)
        My = P*L
        Ugood = np.zeros((2, 6))
        Fgood = np.zeros((2, 6))
        Ugood[1, :3] = uv * v
        Ugood[1, 4] = -ty
        Fgood[0, :3] = -P * v
        Fgood[0, 4] = My
        Fgood[1, :3] = P * v
        np.testing.assert_almost_equal(Utest, Ugood)
        np.testing.assert_almost_equal(Ftest, Fgood)


@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_bendingYtoZ", "test_bendingYtoX"])
def test_bendingYZtoYZ():
    ntests = 10
    for i in range(ntests):
        E = random_between(1, 2)
        nu = random_between(0, 0.49)
        d = random_between(1, 2)
        L = random_between(1, 2)
        P = random_between(-1, 1)
        r = random_unit_vector([False, True, True])
        v = np.cross(r, (1, 0, 0))

        steel = Isotropic(E=E, nu=nu)
        circle = Circle(R=d/2, nu=nu)
        A = (0, 0, 0)
        B = L * r
        bar = EulerBernoulli([A, B])
        bar.material = steel
        bar.section = circle

        U = np.empty((2, 6), dtype="object")
        F = np.zeros((2, 6))
        U[0, :] = 0
        F[1, :3] = P * v

        K = bar.stiffness_matrix()
        Utest, Ftest = solve(K, F, U)

        uv = (64*P*L**3)/(3*np.pi*E*d**4)
        tx = 32*P*L**2/(np.pi*E*d**4)
        Mx = P*L
        Ugood = np.zeros((2, 6))
        Fgood = np.zeros((2, 6))
        Ugood[1, :3] = uv * v
        Ugood[1, 3] = -tx
        Fgood[0, :3] = -P * v
        Fgood[0, 3] = Mx
        Fgood[1, :3] = P * v
        np.testing.assert_almost_equal(Utest, Ugood)
        np.testing.assert_almost_equal(Ftest, Fgood)

@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_bendingXYtoXY", "test_bendingXtoYZ", "test_bendingYtoXZ"])
def test_bendingXYtoXYZ():
    pass

@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_bendingXZtoXZ", "test_bendingXtoYZ", "test_bendingZtoXY"])
def test_bendingXZtoXYZ():
    pass

@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_bendingYZtoYZ", "test_bendingZtoXY", "test_bendingYtoXZ"])
def test_bendingYZtoXYZ():
    pass

@pytest.mark.order(2)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_bendingXYtoXYZ", "test_bendingXZtoXYZ", "test_bendingYZtoXYZ"])
def test_bendingXYZtoXYZ():
    ntests = 10
    for i in range(ntests):
        E = random_between(1, 2)
        nu = random_between(0, 0.49)
        d = random_between(1, 2)
        L = random_between(1, 2)
        P = random_between(-1, 1)
        r = random_unit_vector()
        v = random_vector()
        v = normalize(v-projection(v, r))
        w = np.cross(r, v)
        
        steel = Isotropic(E=E, nu=nu)
        circle = Circle(R=d/2, nu=nu)
        A = (0, 0, 0)
        B = L * r
        bar = EulerBernoulli([A, B])
        bar.material = steel
        bar.section = circle

        U = np.empty((2, 6), dtype="object")
        F = np.zeros((2, 6))
        U[0, :] = 0
        F[1, :3] = P * v

        K = bar.stiffness_matrix()
        Utest, Ftest = solve(K, F, U)

        uv = (64*P*L**3)/(3*np.pi*E*d**4)
        tw = 32*P*L**2/(np.pi*E*d**4)
        Mw = P*L
        Ugood = np.zeros((2, 6))
        Fgood = np.zeros((2, 6))
        Ugood[1, :3] = uv * v
        Ugood[1, 3:] = tw * w
        Fgood[0, :3] = -P * v
        Fgood[0, 3:] = -Mw * w
        Fgood[1, :3] = P * v
        np.testing.assert_almost_equal(Utest, Ugood)
        np.testing.assert_almost_equal(Ftest, Fgood)

@pytest.mark.order(2)
@pytest.mark.dependency(depends=["test_begin", "test_bendingXYZtoXYZ"])
def test_end():
    pass

def main():
    test_begin()
    test_bendingXtoY()
    test_bendingXtoZ()
    test_bendingYtoX()
    test_bendingYtoZ()
    test_bendingZtoX()
    test_bendingZtoY()
    test_bendingXtoYZ()
    test_bendingYtoXZ()
    test_bendingZtoXY()
    test_bendingXYtoXY()
    test_bendingXZtoXZ()
    test_bendingYZtoYZ()
    test_bendingXYtoXYZ()
    test_bendingXZtoXYZ()
    test_bendingYZtoXYZ()
    test_bendingXYZtoXYZ()
    test_end()

if __name__ == "__main__":
    main()