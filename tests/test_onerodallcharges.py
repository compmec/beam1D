import numpy as np
from compmec.strct.material import Isotropic
from compmec.strct.section import Circle
from compmec.strct.element import EulerBernoulli
from compmec.strct.solver import solve
import pytest
from usefulfunc import *

def get_randombase():
    r = np.random.rand(3)
    r /= np.linalg.norm(r)
    v = np.random.rand(3)
    v -= np.sum(r*v) * r
    v /= np.linalg.norm(v)
    w = np.cross(r, v)
    return r, v, w

def compute_U_analitic(stiffvals, force, vectors):
    E, G, L, d = stiffvals
    r, v, w = vectors
    Fr, Mr = force[0], force[3]
    Fv, Mw = force[1], force[5]
    Fw, Mv = force[2], force[4]
    
    Kur = 4*L/(np.pi*E*d**2)
    Ktr = 32*L/(np.pi*G*d**4)
    Kv = 32*L*np.array([[2*L**2/3, L],
                        [L, 2]])/(np.pi*E*d**4)
    Kw = 32*L*np.array([[2*L**2/3, -L],
                        [-L, 2]])/(np.pi*E*d**4)

    ur = Kur * Fr
    tr = Ktr * Mr
    uv, tw = Kv @ (Fv, Mw)
    uw, tv = Kw @ (Fw, Mv)
    R33 = np.array([r, v, w]).T
    Uglo = np.zeros((2, 6), dtype="float64")
    Uglo[1, :3] = R33 @ (ur, uv, uw)
    Uglo[1, 3:] = R33 @ (tr, tv, tw)
    return Uglo

@pytest.mark.order(3)
@pytest.mark.dependency(
    depends=["tests/test_onerodbending.py::test_end",
             "tests/test_onerodtraction.py::test_end",
             "tests/test_onerodtorsion.py::test_end"],
    scope='session'
)
def test_begin():
    pass

@pytest.mark.order(3)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_all():
    ntests = 100
    for i in range(ntests):
        E = random_between(1, 2)
        nu = random_between(0, 0.49)
        G = E / (2*(1+nu))
        d = random_between(1, 2)
        L = random_between(1, 2)
        
        Pall = 2*np.random.rand(6) - 1
        r, v, w = get_randombase()        

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
        F[1, :3] += Pall[0] * r
        F[1, :3] += Pall[1] * v
        F[1, :3] += Pall[2] * w
        F[1, 3:] += Pall[3] * r
        F[1, 3:] += Pall[4] * v
        F[1, 3:] += Pall[5] * w

        K = bar.stiffness_matrix()
        Utest, Ftest = solve(K, F, U)

        Ugood = compute_U_analitic([E, G, L, d], Pall, [r, v, w])
        Fgood = np.zeros((2, 6))
        Fgood[0, :] -= F[1, :]
        Fgood[0, 3:] -= L*np.cross(r, F[1, :3])
        Fgood[1, :] += F[1, :]
        np.testing.assert_almost_equal(Utest, Ugood)
        np.testing.assert_almost_equal(Ftest, Fgood)

@pytest.mark.order(3)
@pytest.mark.dependency(depends=["test_begin", "test_all"])
def test_end():
    pass


def main():
    test_begin()
    test_all()
    test_end()

if __name__ == "__main__":
    main()