from compmec.strct.element import EulerBernoulli
from compmec.strct.material import Isotropic
from compmec.strct.section import Circle
from compmec.strct.system import StaticSystem
import pytest
import numpy as np
from matplotlib import pyplot as plt


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
    A = (0, 0, 0)
    B = (L, 0, 0)
    F0 = 10
    E, d = 210e+3, 8
    beamAB = EulerBernoulli([A, B])
    beamAB.section = Circle(R=d/2, nu=0.3)
    beamAB.material = Isotropic(E=E, nu=0.3)
    for t in np.linspace(0, 1, 17):
        beamAB.addt(t)
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
    A = (0, 0, 0)
    B = (L, 0, 0)
    F0 = 10
    E, d = 210e+3, 8
    I = np.pi*d**4/64

    beamAB = EulerBernoulli([A, B])
    beamAB.section = Circle(R=d/2, nu=0.3)
    beamAB.material = Isotropic(E=E, nu=0.3)
    npts = 17
    ts = np.linspace(0, 1, npts)
    for t in ts:
        beamAB.addt(t)
    system = StaticSystem()
    system.add_element(beamAB)
    system.add_BC(A, {"ux":0,
                      "uy":0,
                      "tz":0})
    system.add_load(B, {"Fy": F0})
    system.run()
    
    uexact = np.zeros((npts, 3))
    Fexact = np.zeros((npts, 3))
    Mexact = np.zeros((npts, 3))
    FEexact = np.zeros((npts, 3))
    MEexact = np.zeros((npts, 3))
    uexact[:, 1] = F0*L**3*ts**2*(3-ts)/(6*E*I)
    Fexact[:, 1] = -F0
    Mexact[:, 2] = F0*L*(ts-1)
    FEexact[0, 1] = -F0
    FEexact[-1, 1] = F0
    MEexact[0, 2] = -F0*L
    ufield = beamAB.field("u")
    Ffield = beamAB.field("F")
    Mfield = beamAB.field("M")
    FEfield = beamAB.field("FE")
    MEfield = beamAB.field("ME")

    np.testing.assert_almost_equal(ufield(ts), uexact)
    # np.testing.assert_almost_equal(Ffield(ts), Fexact)
    # np.testing.assert_almost_equal(Mfield(ts), Mexact)
    # np.testing.assert_almost_equal(MEfield(ts), MEexact)
    # np.testing.assert_almost_equal(FEfield(ts), FEexact)


@pytest.mark.order(9)
@pytest.mark.dependency(depends=["test_begin", "test_rodtraction", "test_cantileverbeam"])
def test_end():
	pass


def main():
    test_begin()
    test_rodtraction()
    test_cantileverbeam()
    test_end()
    
if __name__ == "__main__":
    main()