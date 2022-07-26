import numpy as np
from compmec.strct.material import Isotropic
from compmec.strct.section import Circle
from compmec.strct.element import EulerBernoulli
from compmec.strct.solver import solve
import pytest
from usefulfunc import *

@pytest.mark.dependency(
	depends=["tests/test_material.py::test_end"],
    scope='session'
)
def test_begin():
    pass

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_torsionXcircle():
	ntests = 10
	for i in range(ntests):
		E = random_between(1, 2)
		nu = random_between(0, 0.49)
		d = random_between(1, 2)
		L = random_between(1, 2)
		T = random_between(-1, 1)

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
		F[1, 3] = T

		K = bar.stiffness_matrix()
		Utest, Ftest = solve(K, F, U)

		tx = 64*T*L*(1+nu)/(np.pi*E*d**4)
		Ugood = np.array([[0, 0, 0, 0, 0, 0],
						  [0, 0, 0, tx, 0, 0]])
		Fgood = np.array([[0, 0, 0, -T, 0, 0],
						  [0, 0, 0, T, 0, 0]])

		np.testing.assert_almost_equal(Utest, Ugood)
		np.testing.assert_almost_equal(Ftest, Fgood)	

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin", "test_torsionXcircle"])
def test_torsionYcircle():
	ntests = 10
	for i in range(ntests):
		E = random_between(1, 2)
		nu = random_between(0, 0.49)
		d = random_between(1, 2)
		L = random_between(1, 2)
		T = random_between(-1, 1)

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
		F[1, 4] = T

		K = bar.stiffness_matrix()
		Utest, Ftest = solve(K, F, U)

		ty = 64*T*L*(1+nu)/(np.pi*E*d**4)
		Ugood = np.array([[0, 0, 0, 0, 0, 0],
						  [0, 0, 0, 0, ty, 0]])
		Fgood = np.array([[0, 0, 0, 0, -T, 0],
						  [0, 0, 0, 0, T, 0]])

		np.testing.assert_almost_equal(Utest, Ugood)
		np.testing.assert_almost_equal(Ftest, Fgood)

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin", "test_torsionXcircle"])
def test_torsionZcircle():
	ntests = 10
	for i in range(ntests):
		E = random_between(1, 2)
		nu = random_between(0, 0.49)
		d = random_between(1, 2)
		L = random_between(1, 2)
		T = random_between(-1, 1)

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
		F[1, 5] = T

		K = bar.stiffness_matrix()
		Utest, Ftest = solve(K, F, U)

		tz = 64*T*L*(1+nu)/(np.pi*E*d**4)
		Ugood = np.array([[0, 0, 0, 0, 0, 0],
						  [0, 0, 0, 0, 0, tz]])
		Fgood = np.array([[0, 0, 0, 0, 0, -T],
						  [0, 0, 0, 0, 0, T]])

		np.testing.assert_almost_equal(Utest, Ugood)
		np.testing.assert_almost_equal(Ftest, Fgood)

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_torsionXcircle", "test_torsionYcircle"])
def test_torsionXYcircle():
	ntests = 10
	for i in range(ntests):
		E = random_between(1, 2)
		nu = random_between(0, 0.49)
		d = random_between(1, 2)
		L = random_between(1, 2)
		T = random_between(-1, 1)
		r = random_unit_vector([True, True, False])
		
		steel = Isotropic(E=E, nu=nu)
		circle = Circle(R=d/2, nu=nu)
		A = (0, 0, 0)
		B = L*r
		bar = EulerBernoulli([A, B])
		bar.material = steel
		bar.section = circle

		U = np.empty((2, 6), dtype="object")
		F = np.zeros((2, 6))
		U[0, :] = 0
		F[1, 3:] = T * r

		K = bar.stiffness_matrix()
		Utest, Ftest = solve(K, F, U)

		tr = 64*T*L*(1+nu)/(np.pi*E*d**4)
		Ugood = np.zeros((2, 6))
		Fgood = np.zeros((2, 6))
		Ugood[1, 3:] = tr * r
		Fgood[0, 3:] = -T * r
		Fgood[1, 3:] = T * r

		np.testing.assert_almost_equal(Utest, Ugood)
		np.testing.assert_almost_equal(Ftest, Fgood)

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_torsionYcircle", "test_torsionZcircle"])
def test_torsionYZcircle():
	ntests = 10
	for i in range(ntests):
		E = random_between(1, 2)
		nu = random_between(0, 0.49)
		d = random_between(1, 2)
		L = random_between(1, 2)
		T = random_between(-1, 1)
		r = random_unit_vector([False, True, True])

		steel = Isotropic(E=E, nu=nu)
		circle = Circle(R=d/2, nu=nu)
		A = (0, 0, 0)
		B = L*r
		bar = EulerBernoulli([A, B])
		bar.material = steel
		bar.section = circle

		U = np.empty((2, 6), dtype="object")
		F = np.zeros((2, 6))
		U[0, :] = 0
		F[1, 3:] = T * r

		K = bar.stiffness_matrix()
		Utest, Ftest = solve(K, F, U)

		tr = 64*T*L*(1+nu)/(np.pi*E*d**4)
		Ugood = np.zeros((2, 6))
		Fgood = np.zeros((2, 6))
		Ugood[1, 3:] = tr * r
		Fgood[0, 3:] = -T * r
		Fgood[1, 3:] = T * r

		np.testing.assert_almost_equal(Utest, Ugood)
		np.testing.assert_almost_equal(Ftest, Fgood)

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_torsionXcircle", "test_torsionZcircle"])
def test_torsionXZcircle():
	ntests = 10
	for i in range(ntests):
		E = random_between(1, 2)
		nu = random_between(0, 0.49)
		d = random_between(1, 2)
		L = random_between(1, 2)
		T = random_between(-1, 1)
		r = random_unit_vector([True, False, True])

		steel = Isotropic(E=E, nu=nu)
		circle = Circle(R=d/2, nu=nu)
		A = (0, 0, 0)
		B = L*r
		bar = EulerBernoulli([A, B])
		bar.material = steel
		bar.section = circle

		U = np.empty((2, 6), dtype="object")
		F = np.zeros((2, 6))
		U[0, :] = 0
		F[1, 3:] = T * r

		K = bar.stiffness_matrix()
		Utest, Ftest = solve(K, F, U)

		tr = 64*T*L*(1+nu)/(np.pi*E*d**4)
		Ugood = np.zeros((2, 6))
		Fgood = np.zeros((2, 6))
		Ugood[1, 3:] = tr * r
		Fgood[0, 3:] = -T * r
		Fgood[1, 3:] = T * r

		np.testing.assert_almost_equal(Utest, Ugood)
		np.testing.assert_almost_equal(Ftest, Fgood)

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_torsionXYcircle", "test_torsionYZcircle", "test_torsionXZcircle"])
def test_torsionRcircle():
	ntests = 10
	for i in range(ntests):
		r = random_unit_vector()
		E = random_between(1, 2)
		nu = random_between(0, 0.49)
		d = random_between(1, 2)
		L = random_between(1, 2)
		T = random_between(-1, 1)

		steel = Isotropic(E=E, nu=nu)
		circle = Circle(R=d/2, nu=nu)
		A = (0, 0, 0)
		B = L*r
		bar = EulerBernoulli([A, B])
		bar.material = steel
		bar.section = circle

		U = np.empty((2, 6), dtype="object")
		F = np.zeros((2, 6))
		U[0, :] = 0
		F[1, 3:] = T * r

		K = bar.stiffness_matrix()
		Utest, Ftest = solve(K, F, U)

		tr = 64*T*L*(1+nu)/(np.pi*E*d**4)
		Ugood = np.zeros((2, 6))
		Fgood = np.zeros((2, 6))
		Ugood[1, 3:] = tr * r
		Fgood[0, 3:] = -T * r
		Fgood[1, 3:] = T * r

		np.testing.assert_almost_equal(Utest, Ugood)
		np.testing.assert_almost_equal(Ftest, Fgood)

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin", "test_torsionRcircle"])
def test_end():
	pass

if __name__ == "__main__":
	pytest.main()