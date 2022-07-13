import pytest
import numpy as np
from compmec.strct.beam import EulerBernoulli
from compmec.strct.material import Isotropic
from compmec.strct.section import Circle
from compmec.strct.system import StaticSystem



def test_example1():
    A = (0, 0)
    B = (1000, 0)
    beamAB = EulerBernoulli([A, B])
    beamAB.section = Circle(R=8/2, nu=0.3)
    beamAB.material = Isotropic(E=210e+3, nu=0.3)
    system = StaticSystem()
    system.add_element(beamAB)
    system.add_BC(A, {"ux":0,
                    "uy":0,
                    "tz":0})
    system.add_load(B, {"Fx": 10})
    system.run()
    Usolu = system.solution
    Ugood = np.zeros((2, 6))
    Ugood[1, 0] = 4*10*1000/(210e+3 * np.pi*8**2 )
    np.testing.assert_almost_equal(Usolu, Ugood)


def test_example2():
    A = (0, 0)
    B = (1000, 0)
    beamAB = EulerBernoulli([A, B])
    beamAB.section = Circle(R=8/2, nu=0.3)
    beamAB.material = Isotropic(E=210e+3, nu=0.3)
    system = StaticSystem()
    system.add_element(beamAB)
    system.add_BC(A, {"ux": 0,
                      "uy": 0,
                      "tz": 0})
    system.add_load(B, {"Fy": -10})
    system.run()
    Usolu = system.solution
    Ugood = np.zeros((2, 6))
    Ugood[1, 1] = -64*10*(1000**3)/(3*np.pi* 210e+3 * 8**4)
    Ugood[1, 5] = -32*10*(1000**2)/(np.pi* 210e+3 * 8**4)
    np.testing.assert_almost_equal(Usolu, Ugood)

if __name__ == "__main__":
    pytest.main()
