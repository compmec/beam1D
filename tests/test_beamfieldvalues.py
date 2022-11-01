from compmec.strct.element import EulerBernoulli
from compmec.strct.material import Isotropic
from compmec.strct.section import Circle
from compmec.strct.system import StaticSystem
import pytest
import numpy as np


@pytest.mark.order(9)
@pytest.mark.dependency(
	depends=["tests/test_onerodallcharges.py::test_end",
             "tests/test_bendingretangular.py::test_end"],
    scope='session'
)
def test_begin():
	pass

@pytest.mark.order(9)
@pytest.mark.timeout(3)
@pytest.mark.dependency(depends=["test_begin"])
def test_rodtraction():
    L = 1000
    A = (0, 0)
    B = (L, 0)
    F0 = 10
    E, d = 210e+3, 8
    beamAB = EulerBernoulli([A, B])
    beamAB.section = Circle(R=d/2, nu=0.3)
    beamAB.material = Isotropic(E=E, nu=0.3)
    for t in np.linspace(0, 1, 17):
        beamAB.path(t)
    system = StaticSystem()
    system.add_element(beamAB)
    system.add_BC(A, {"ux":0,
                      "uy":0,
                      "tz":0})
    system.add_load(B, {"Fx": F0})
    system.run()

    npts = 129
    t = np.linspace(0, 1, npts)
    A = np.pi * (d/2)**2
    uexact = np.zeros((npts, 3))
    uexact[:, 0] = F0*L*t/(E * A)
    ufield = beamAB.field("u")
    uguess = ufield(t)
    np.testing.assert_almost_equal(uguess, uexact)

@pytest.mark.order(9)
@pytest.mark.timeout(3)
@pytest.mark.dependency(depends=["test_begin"])
def test_cantileverbeam():
    L = 1000
    A = (0, 0)
    B = (L, 0)
    F0 = 10
    E, d = 210e+3, 8
    beamAB = EulerBernoulli([A, B])
    beamAB.section = Circle(R=d/2, nu=0.3)
    beamAB.material = Isotropic(E=E, nu=0.3)
    for t in np.linspace(0, 1, 17):
        beamAB.path(t)
    system = StaticSystem()
    system.add_element(beamAB)
    system.add_BC(A, {"ux":0,
                      "uy":0,
                      "tz":0})
    system.add_load(B, {"Fy": F0})
    system.run()

    npts = 129
    t = np.linspace(0, 1, npts)
    I = np.pi * d**4 / 64
    uexact = np.zeros((npts, 3))
    uexact[:, 1] = F0*L**3*t**2*(3-t)/(6*E*I)
    ufield = beamAB.field("u")
    np.testing.assert_almost_equal(ufield(t), uexact)


@pytest.mark.order(9)
@pytest.mark.dependency(depends=["test_begin", "test_rodtraction", "test_cantileverbeam"])
def test_end():
	pass

def main():
    test_begin()
    test_rodtraction()
    test_end()
    
if __name__ == "__main__":
    main()