import numpy as np
from compmec.strct.material import Isotropic
from compmec.strct.section import Circle
from compmec.strct.beam import EulerBernoulli
from compmec.strct.solver import solve
import pytest

@pytest.mark.dependency()
def test_begin():
    pass

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_bendingXtoY():
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
		bar = EulerBernoulli([A, B])
		bar.material = steel
		bar.section = circle

		U = np.empty((2, 6), dtype="object")
		F = np.zeros((2, 6))

		# U[0, 0] = 0
		U[0, :] = 0
		F[1, 1] = P

		K = bar.stiffness_matrix()
		Utest, Ftest = solve(K, F, U)


		uy = (64*P*L**3)/(3*np.pi*E*d**4)
		tz = 32*P*L**2/(np.pi*E*d**4)
		Mz = P*L
		Ugood = np.array([[0, 0, 0, 0, 0, 0],
						  [0, uy, 0, 0, 0, tz]])
		Fgood = np.array([[0, -P, 0, 0, 0, -Mz],
						  [0, P, 0, 0, 0, 0]])
		np.testing.assert_almost_equal(Utest, Ugood)
		np.testing.assert_almost_equal(Ftest, Fgood)
		
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_bendingXtoZ():
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
		bar = EulerBernoulli([A, B])
		bar.material = steel
		bar.section = circle

		U = np.empty((2, 6), dtype="object")
		F = np.zeros((2, 6))

		# U[0, 0] = 0
		U[0, :] = 0
		F[1, 2] = P

		K = bar.stiffness_matrix()
		Utest, Ftest = solve(K, F, U)


		uz = (64*P*L**3)/(3*np.pi*E*d**4)
		ty = 32*P*L**2/(np.pi*E*d**4)
		My = P*L
		Ugood = np.array([[0, 0, 0, 0, 0, 0],
						  [0, 0, uz, 0, -ty, 0]])
		Fgood = np.array([[0, 0, -P, 0, My, 0],
						  [0, 0, P, 0, 0, 0]])
		np.testing.assert_almost_equal(Utest, Ugood)
		np.testing.assert_almost_equal(Ftest, Fgood)
		

# @pytest.mark.timeout(2)
# @pytest.mark.dependency(depends=["test_tractionR", "test_torsionRcircle"])
# def test_end():
# 	pass

if __name__ == "__main__":
	pytest.main()
	# main()