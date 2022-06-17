import numpy as np
from compmec.strct.material import Isotropic
from compmec.strct.section import Circle
from compmec.strct.beam import EulerBernoulli
from compmec.strct.solver import solve
import pytest

@pytest.mark.dependency()
def test_begin():
    pass

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_bendingXtoY():
    ntests = 10
    for i in range(ntests):
        E = 1 + np.random.rand()
        nu = 0.49*np.random.rand()
        d = 1 + np.random.rand()
        L = 1 + np.random.rand()
        P = 2*np.random.rand()-1

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
        
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_bendingXtoZ():
    ntests = 10
    for i in range(ntests):
        E = 1 + np.random.rand()
        nu = 0.49*np.random.rand()
        d = 1 + np.random.rand()
        L = 1 + np.random.rand()
        P = 2*np.random.rand()-1

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


@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_bendingXtoY", "test_bendingXtoZ"])
def test_bendingXtoYZ():
    ntests = 10
    for i in range(ntests):
        E = 1 + np.random.rand()
        nu = 0.49*np.random.rand()
        d = 1 + np.random.rand()
        L = 1 + np.random.rand()
        P = 2*np.random.rand()-1
        theta = 2*np.pi*np.random.rand()
        c, s = np.cos(theta), np.sin(theta)
        r = np.array([0, c, s])
        rx = np.array([0, -s, c])

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
        Ugood[1, 3:] = tr * rx
        Fgood[0, :3] = -P * r
        Fgood[0, 3:] = -Mr * rx
        Fgood[1, :3] = P * r
        
        np.testing.assert_almost_equal(Utest, Ugood)
        np.testing.assert_almost_equal(Ftest, Fgood)


@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_bendingYtoX():
    ntests = 10
    for i in range(ntests):
        E = 1 + np.random.rand()
        nu = 0.49*np.random.rand()
        d = 1 + np.random.rand()
        L = 1 + np.random.rand()
        P = 2*np.random.rand()-1

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
        
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_bendingYtoZ():
    ntests = 10
    for i in range(ntests):
        E = 1 + np.random.rand()
        nu = 0.49*np.random.rand()
        d = 1 + np.random.rand()
        L = 1 + np.random.rand()
        P = 2*np.random.rand()-1

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


@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_bendingYtoX", "test_bendingYtoZ"])
def test_bendingYtoXZ():
    ntests = 10
    for i in range(ntests):
        E = 1 + np.random.rand()
        nu = 0.49*np.random.rand()
        d = 1 + np.random.rand()
        L = 1 + np.random.rand()
        P = 2*np.random.rand()-1
        theta = 2*np.pi*np.random.rand()
        c, s = np.cos(theta), np.sin(theta)
        r = np.array([c, 0, s])
        rx = np.array([s, 0, -c])

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
        Ugood[1, 3:] = tr * rx
        Fgood[0, :3] = -P * r
        Fgood[0, 3:] = -Mr * rx
        Fgood[1, :3] = P * r
        
        np.testing.assert_almost_equal(Utest, Ugood)
        np.testing.assert_almost_equal(Ftest, Fgood)
        
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_bendingZtoX():
    ntests = 10
    for i in range(ntests):
        E = 1 + np.random.rand()
        nu = 0.49*np.random.rand()
        d = 1 + np.random.rand()
        L = 1 + np.random.rand()
        P = 2*np.random.rand()-1

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
        
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_bendingZtoY():
    ntests = 10
    for i in range(ntests):
        E = 1 + np.random.rand()
        nu = 0.49*np.random.rand()
        d = 1 + np.random.rand()
        L = 1 + np.random.rand()
        P = 2*np.random.rand()-1

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


@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_bendingZtoX", "test_bendingZtoY"])
def test_bendingZtoXY():
    ntests = 10
    for i in range(ntests):
        E = 1 + np.random.rand()
        nu = 0.49*np.random.rand()
        d = 1 + np.random.rand()
        L = 1 + np.random.rand()
        P = 2*np.random.rand()-1
        theta = 2*np.pi*np.random.rand()
        c, s = np.cos(theta), np.sin(theta)
        r = np.array([c, s, 0])
        rx = np.array([-s, c, 0])

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
        Ugood[1, 3:] = tr * rx
        Fgood[0, :3] = -P * r
        Fgood[0, 3:] = -Mr * rx
        Fgood[1, :3] = P * r
        
        np.testing.assert_almost_equal(Utest, Ugood)
        np.testing.assert_almost_equal(Ftest, Fgood)


@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_bendingXtoY", "test_bendingYtoX"])
def test_bendingXYtoXY():
    ntests = 10
    for i in range(ntests):
        E = 1 + np.random.rand()
        nu = 0.49*np.random.rand()
        d = 1 + np.random.rand()
        L = 1 + np.random.rand()
        P = 2*np.random.rand()-1
        theta = 2*np.pi*np.random.rand()
        c, s = np.cos(theta), np.sin(theta)
        r = np.array([c, s, 0])
        v = np.array([-s, c, 0])

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
        Ugood[1, 5] = tz
        Fgood[0, :3] = -P * v
        Fgood[0, 5] = -Mz
        Fgood[1, :3] = P * v
        np.testing.assert_almost_equal(Utest, Ugood)
        np.testing.assert_almost_equal(Ftest, Fgood)

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_bendingXtoZ", "test_bendingZtoX"])
def test_bendingXZtoXZ():
    ntests = 10
    for i in range(ntests):
        E = 1 + np.random.rand()
        nu = 0.49*np.random.rand()
        d = 1 + np.random.rand()
        L = 1 + np.random.rand()
        P = 2*np.random.rand()-1
        theta = 2*np.pi*np.random.rand()
        c, s = np.cos(theta), np.sin(theta)
        r = np.array([c, 0, s])
        v = np.array([s, 0, -c])

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
        Ugood[1, 4] = ty
        Fgood[0, :3] = -P * v
        Fgood[0, 4] = -My
        Fgood[1, :3] = P * v
        np.testing.assert_almost_equal(Utest, Ugood)
        np.testing.assert_almost_equal(Ftest, Fgood)


@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_bendingYtoZ", "test_bendingYtoX"])
def test_bendingYZtoYZ():
    ntests = 10
    for i in range(ntests):
        E = 1 + np.random.rand()
        nu = 0.49*np.random.rand()
        d = 1 + np.random.rand()
        L = 1 + np.random.rand()
        P = 2*np.random.rand()-1
        theta = 2*np.pi*np.random.rand()
        c, s = np.cos(theta), np.sin(theta)
        r = np.array([0, c, s])
        v = np.array([0, s, -c])

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


@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_bendingXYtoXY", "test_bendingXtoYZ", "test_bendingYtoXZ"])
def test_bendingXYtoXYZ():
    pass

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_bendingXZtoXZ", "test_bendingXtoYZ", "test_bendingZtoXY"])
def test_bendingXZtoXYZ():
    pass

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_bendingYZtoYZ", "test_bendingZtoXY", "test_bendingYtoXZ"])
def test_bendingYZtoXYZ():
    pass

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_bendingXYtoXYZ", "test_bendingXZtoXYZ", "test_bendingYZtoXYZ"])
def test_bendingXYZtoXYZ():
    ntests = 10
    for i in range(ntests):
        E = 1 + np.random.rand()
        nu = 0.49*np.random.rand()
        d = 1 + np.random.rand()
        L = 1 + np.random.rand()
        P = 2*np.random.rand()-1
        theta = 2*np.pi*np.random.rand()
        r = np.random.rand(3)
        r /= np.linalg.norm(r)
        v = np.random.rand(3)
        v -= np.sum(r*v) * r
        v /= np.linalg.norm(v)
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

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_bendingXYZtoXYZ"])
def test_end():
    pass

if __name__ == "__main__":
    pytest.main()