import numpy as np
from matplotlib import pyplot as plt

from compmec.strct.element import EulerBernoulli
from compmec.strct.material import Isotropic
from compmec.strct.profile import Circle
from compmec.strct.system import StaticSystem

lenght = 1000
A = (0, 0, 0)
B = (lenght, 0, 0)
beam = EulerBernoulli([A, B])

E = 210e3
nu = 0.3
d = 8
profile = Circle(R=d / 2)
material = Isotropic(E=E, nu=nu)
beam.section = material, profile

system = StaticSystem()
system.add_element(beam)
boundary_conditions = {"ux": 0, "uy": 0, "tz": 0}
system.add_BC(A, boundary_conditions)

applied_force = 10
system.add_load(B, {"Fy": applied_force})

npts = 101
ts = np.linspace(0, 1, npts)
beam.path.knot_insert(ts)
system.run()

curve = beam.field("u")  # Displacement curve
values_test = curve(ts)
values_good = np.zeros((npts, 3))
values_good[:, 1] = 64 * applied_force * (ts * lenght) ** 3
values_good[:, 1] /= 3 * E * np.pi * d**4


plt.plot(ts, values_test[:, 1], label="test")
plt.plot(ts, values_good[:, 1], label="good")
plt.legend()
plt.show()
