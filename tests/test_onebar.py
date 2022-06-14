import numpy as np
from material import Isotropic
from section import Circle
from beam import EulerBernoulli
from solver import solve
import pytest

@pytest.mark.timeout(2)
def test_tractionX():
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
		bar = EulerBernoulli(A, B)
		bar.material = steel
		bar.section = circle

		U = np.empty((2, 6), dtype="object")
		F = np.zeros((2, 6))

		# U[0, 0] = 0
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
def test_tractionY():
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
		bar = EulerBernoulli(A, B)
		bar.material = steel
		bar.section = circle

		U = np.empty((2, 6), dtype="object")
		F = np.zeros((2, 6))

		# U[0, 1] = 0
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
def test_tractionZ():
	ntests = 10
	for i in range(ntests):
		E = 1 + np.random.rand()
		nu = 0.49*np.random.rand()
		d = 1 + np.random.rand()
		L = 2*np.random.rand()-1
		P = 2*np.random.rand()-1

		steel = Isotropic(E=E, nu=nu)
		circle = Circle(R=d/2, nu=nu)
		A = (0, 0, 0)
		B = (0, 0, L)
		bar = EulerBernoulli(A, B)
		bar.material = steel
		bar.section = circle

		U = np.empty((2, 6), dtype="object")
		F = np.zeros((2, 6))

		# U[0, 2] = 0
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
def test_tractionR():
	ntests = 10
	for i in range(ntests):
		r = np.random.rand(3)  # random direction
		r = np.array([1, 1, 0], dtype="float64")
		r /= np.linalg.norm(r)  # To r a unit vector  
		E = 1 + np.random.rand()
		nu = 0.49*np.random.rand()
		d = 1 + np.random.rand()
		L = 2*np.random.rand()-1
		P = 2*np.random.rand()-1

		steel = Isotropic(E=E, nu=nu)
		circle = Circle(R=d/2, nu=nu)
		A = (0, 0, 0)
		B = L*r
		bar = EulerBernoulli(A, B)
		bar.material = steel
		bar.section = circle

		U = np.empty((2, 6), dtype="object")
		F = np.zeros((2, 6))

		# U[0, :3] = 0
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


if __name__ == "__main__":
	pytest.main()
		