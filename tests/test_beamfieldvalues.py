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
@pytest.mark.timeout(2)
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
    for t in np.linspace(0, 1, 65):
        beamAB.path(t)
    system = StaticSystem()
    system.add_element(beamAB)
    system.add_BC(A, {"ux":0,
                    "uy":0,
                    "tz":0})
    system.add_load(B, {"Fx": F0})
    system.run()

    t = np.linspace(0, 1, 129)
    A = np.pi * (d/2)**2
    uxexact = F0*L*t/(E * A)
    uxfield = beamAB.field("ux")(t).reshape(-1)
    uyfield = beamAB.field("uy")(t).reshape(-1)
    uzfield = beamAB.field("uz")(t).reshape(-1)
    np.testing.assert_almost_equal(uxfield, uxexact)
    np.testing.assert_almost_equal(uyfield, np.zeros(uyfield.shape))
    np.testing.assert_almost_equal(uzfield, np.zeros(uzfield.shape))

@pytest.mark.order(9)
@pytest.mark.dependency(depends=["test_begin", "test_rodtraction"])
def test_end():
	pass

def main():
    test_begin()
    test_rodtraction()
    test_end()
    
if __name__ == "__main__":
    main()