from compmec.strct.element import EulerBernoulli
from compmec.strct.profile import Circle, Square
from compmec.strct.material import Isotropic
from compmec.strct.system import StaticSystem
from compmec.strct.shower import ShowerStaticSystem
from matplotlib import pyplot as plt
import numpy as np


def tst_example09():
    pass

def tst_example20():
    A = (0, 0, 0)
    B = (1000, 500, 0)
    C = (1000, -500, 0)
    D = (1000, 0, 500)
    beamAB = EulerBernoulli([A, B])
    beamAC = EulerBernoulli([A, C])
    beamAD = EulerBernoulli([A, D])
    beamBC = EulerBernoulli([B, C])
    beamBD = EulerBernoulli([B, D])
    beamCD = EulerBernoulli([C, D])
    circle = Circle(R=8/2, nu=0.3)
    steel = Isotropic(E=210e+3, nu=0.3)
    system = StaticSystem()
    for beam in [beamAB, beamAC, beamAD, beamBC, beamBD, beamCD]:
        for t in np.linspace(0, 1, 21):
            beam.path(t)
        beam.material = steel
        beam.section = circle
        system.add_element(beam)
    E = beamBC.path(0.5)
    system.add_BC(A, {"ux": 0,
                    "uy": 0,
                    "uz": 0,
                    "tx": 0,
                    "ty": 0,
                    "tz": 0})
    system.add_BC(B, {"uz": 0})
    system.add_BC(C, {"uz": 0})
    system.add_load(D, {"Fz": 5000000})
    system.run()

    shower = ShowerStaticSystem(system)

    plt.figure()
    ax = plt.gca()
    shower.plot2D(projector='xy', deformed=False, axes=ax)
    shower.plot2D(projector='xy', deformed=True, axes=ax)
    plt.figure()
    ax = plt.axes(projection='3d')
    shower.plot3D(deformed=False, axes=ax)
    shower.plot3D(deformed=True, axes=ax)
    plt.legend()
    plt.show()


def main():
    tst_example20()

if __name__ == "__main__":
    main()