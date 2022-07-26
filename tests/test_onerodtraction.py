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
def test_tractionX():
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
		F[1, 0] = P

		K = bar.stiffness_matrix()
		Utest, Ftest = solve(K, F, U)

		ux = (4*P*L)/(np.pi*d**2*E)
		Ugood = np.array([[0, 0, 0, 0, 0, 0],
						  [ux, 0, 0, 0, 0, 0]])
		Fgood = np.array([[-P, 0, 0, 0, 0, 0],
						  [P, 0, 0, 0, 0, 0]])

		np.testing.assert_almost_equal(Utest, Ugood)
		np.testing.assert_almost_equal(Ftest, Fgood)
		
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin", "test_tractionX"])
def test_tractionY():
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
		F[1, 1] = P

		K = bar.stiffness_matrix()
		Utest, Ftest = solve(K, F, U)

		uy = (4*P*L)/(np.pi*d**2*E)
		Ugood = np.array([[0, 0, 0, 0, 0, 0],
						  [0, uy, 0, 0, 0, 0]])
		Fgood = np.array([[0, -P, 0, 0, 0, 0],
						  [0, P, 0, 0, 0, 0]])

		np.testing.assert_almost_equal(Utest, Ugood)
		np.testing.assert_almost_equal(Ftest, Fgood)

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin", "test_tractionX"])
def test_tractionZ():
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
		F[1, 2] = P

		K = bar.stiffness_matrix()
		Utest, Ftest = solve(K, F, U)

		uz = (4*P*L)/(np.pi*d**2*E)
		Ugood = np.array([[0, 0, 0, 0, 0, 0],
						  [0, 0, uz, 0, 0, 0]])
		Fgood = np.array([[0, 0, -P, 0, 0, 0],
						  [0, 0, P, 0, 0, 0]])
		np.testing.assert_almost_equal(Utest, Ugood)
		np.testing.assert_almost_equal(Ftest, Fgood)

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_tractionX", "test_tractionY"])
def test_tractionXY():
	ntests = 10
	for i in range(ntests):
		E = random_between(1, 2)
		nu = random_between(0, 0.49)
		d = random_between(1, 2)
		L = random_between(1, 2)
		P = random_between(-1, 1)
		r = random_unit_vector([True, True, False])

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
		F[1, :3] = P * r

		K = bar.stiffness_matrix()
		Utest, Ftest = solve(K, F, U)

		ur = (4*P*L)/(np.pi*d**2*E)
		Ugood = np.zeros((2, 6))
		Fgood = np.zeros((2, 6))
		Ugood[1, :3] = ur * r
		Fgood[0, :3] = -P * r
		Fgood[1, :3] = P * r

		np.testing.assert_almost_equal(Utest, Ugood)
		np.testing.assert_almost_equal(Ftest, Fgood)

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_tractionY", "test_tractionZ"])
def test_tractionYZ():
	ntests = 10
	for i in range(ntests):
		E = random_between(1, 2)
		nu = random_between(0, 0.49)
		d = random_between(1, 2)
		L = random_between(1, 2)
		P = random_between(-1, 1)
		r = random_unit_vector([False, True, True])

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
		F[1, :3] = P * r

		K = bar.stiffness_matrix()
		Utest, Ftest = solve(K, F, U)

		ur = (4*P*L)/(np.pi*d**2*E)
		Ugood = np.zeros((2, 6))
		Fgood = np.zeros((2, 6))
		Ugood[1, :3] = ur * r
		Fgood[0, :3] = -P * r
		Fgood[1, :3] = P * r

		np.testing.assert_almost_equal(Utest, Ugood)
		np.testing.assert_almost_equal(Ftest, Fgood)

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_tractionX", "test_tractionZ"])
def test_tractionXZ():
	ntests = 10
	for i in range(ntests):
		E = random_between(1, 2)
		nu = random_between(0, 0.49)
		d = random_between(1, 2)
		L = random_between(1, 2)
		P = random_between(-1, 1)
		r = random_unit_vector([True, False, True])

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
		F[1, :3] = P * r

		K = bar.stiffness_matrix()
		Utest, Ftest = solve(K, F, U)

		ur = (4*P*L)/(np.pi*d**2*E)
		Ugood = np.zeros((2, 6))
		Fgood = np.zeros((2, 6))
		Ugood[1, :3] = ur * r
		Fgood[0, :3] = -P * r
		Fgood[1, :3] = P * r

		np.testing.assert_almost_equal(Utest, Ugood)
		np.testing.assert_almost_equal(Ftest, Fgood)

		
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_tractionXY", "test_tractionYZ", "test_tractionXZ"])
def test_tractionR():
	ntests = 10
	for i in range(ntests):
		r = random_unit_vector()
		E = random_between(1, 2)
		nu = random_between(0, 0.49)
		d = random_between(1, 2)
		L = random_between(1, 2)
		P = random_between(-1, 1)

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
		F[1, :3] = P * r

		K = bar.stiffness_matrix()
		Utest, Ftest = solve(K, F, U)

		ur = (4*P*L)/(np.pi*d**2*E)
		Ugood = np.zeros((2, 6))
		Fgood = np.zeros((2, 6))
		Ugood[1, :3] = ur * r
		Fgood[0, :3] = -P * r
		Fgood[1, :3] = P * r

		np.testing.assert_almost_equal(Utest, Ugood)
		np.testing.assert_almost_equal(Ftest, Fgood)

@pytest.mark.dependency(depends=["test_begin", "test_tractionR"])
def test_end():
	pass

if __name__ == "__main__":
	pytest.main()