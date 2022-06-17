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
		bar = EulerBernoulli([A, B])
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
@pytest.mark.dependency(depends=["test_begin"])
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
		bar = EulerBernoulli([A, B])
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
@pytest.mark.dependency(depends=["test_begin"])
def test_tractionZ():
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
		B = (0, 0, L)
		bar = EulerBernoulli([A, B])
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
@pytest.mark.dependency(depends=["test_tractionX", "test_tractionY"])
def test_tractionXY():
	ntests = 10
	for i in range(ntests):
		E = 1 + np.random.rand()
		nu = 0.49*np.random.rand()
		d = 1 + np.random.rand()
		L = 1 + np.random.rand()
		P = 2*np.random.rand()-1
		rx, ry = np.random.rand(2)
		r = np.array([rx, ry, 0])
		r /= np.linalg.norm(r)

		steel = Isotropic(E=E, nu=nu)
		circle = Circle(R=d/2, nu=nu)
		A = (0, 0, 0)
		B = L * r
		bar = EulerBernoulli([A, B])
		bar.material = steel
		bar.section = circle

		U = np.empty((2, 6), dtype="object")
		F = np.zeros((2, 6))

		# U[0, 0] = 0
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
		E = 1 + np.random.rand()
		nu = 0.49*np.random.rand()
		d = 1 + np.random.rand()
		L = 1 + np.random.rand()
		P = 2*np.random.rand()-1
		ry, rz = np.random.rand(2)
		r = np.array([0, ry, rz])
		r /= np.linalg.norm(r)

		steel = Isotropic(E=E, nu=nu)
		circle = Circle(R=d/2, nu=nu)
		A = (0, 0, 0)
		B = L * r
		bar = EulerBernoulli([A, B])
		bar.material = steel
		bar.section = circle

		U = np.empty((2, 6), dtype="object")
		F = np.zeros((2, 6))

		# U[0, 0] = 0
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
		E = 1 + np.random.rand()
		nu = 0.49*np.random.rand()
		d = 1 + np.random.rand()
		L = 1 + np.random.rand()
		P = 2*np.random.rand()-1
		rx, rz = np.random.rand(2)
		r = np.array([rx, 0, rz])
		r /= np.linalg.norm(r)

		steel = Isotropic(E=E, nu=nu)
		circle = Circle(R=d/2, nu=nu)
		A = (0, 0, 0)
		B = L * r
		bar = EulerBernoulli([A, B])
		bar.material = steel
		bar.section = circle

		U = np.empty((2, 6), dtype="object")
		F = np.zeros((2, 6))

		# U[0, 0] = 0
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
		r = np.random.rand(3)  # random direction
		r /= np.linalg.norm(r)  # To r a unit vector  
		E = 1 + np.random.rand()
		nu = 0.49*np.random.rand()
		d = 1 + np.random.rand()
		L = 1 + np.random.rand()
		P = 2*np.random.rand()-1

		steel = Isotropic(E=E, nu=nu)
		circle = Circle(R=d/2, nu=nu)
		A = (0, 0, 0)
		B = L*r
		bar = EulerBernoulli([A, B])
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

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_torsionXcircle():
	ntests = 10
	for i in range(ntests):
		E = 1 + np.random.rand()
		nu = 0.49*np.random.rand()
		d = 1 + np.random.rand()
		L = 1 + np.random.rand()
		T = 2*np.random.rand()-1

		steel = Isotropic(E=E, nu=nu)
		circle = Circle(R=d/2, nu=nu)
		A = (0, 0, 0)
		B = (L, 0, 0)
		bar = EulerBernoulli([A, B])
		bar.material = steel
		bar.section = circle

		U = np.empty((2, 6), dtype="object")
		F = np.zeros((2, 6))

		# U[0, 3] = 0
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
@pytest.mark.dependency(depends=["test_begin"])
def test_torsionYcircle():
	ntests = 10
	for i in range(ntests):
		E = 1 + np.random.rand()
		nu = 0.49*np.random.rand()
		d = 1 + np.random.rand()
		L = 1 + np.random.rand()
		T = 2*np.random.rand()-1

		steel = Isotropic(E=E, nu=nu)
		circle = Circle(R=d/2, nu=nu)
		A = (0, 0, 0)
		B = (0, L, 0)
		bar = EulerBernoulli([A, B])
		bar.material = steel
		bar.section = circle

		U = np.empty((2, 6), dtype="object")
		F = np.zeros((2, 6))

		# U[0, 4] = 0
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
@pytest.mark.dependency(depends=["test_begin"])
def test_torsionZcircle():
	ntests = 10
	for i in range(ntests):
		E = 1 + np.random.rand()
		nu = 0.49*np.random.rand()
		d = 1 + np.random.rand()
		L = 1 + np.random.rand()
		T = 2*np.random.rand()-1

		steel = Isotropic(E=E, nu=nu)
		circle = Circle(R=d/2, nu=nu)
		A = (0, 0, 0)
		B = (0, 0, L)
		bar = EulerBernoulli([A, B])
		bar.material = steel
		bar.section = circle

		U = np.empty((2, 6), dtype="object")
		F = np.zeros((2, 6))

		# U[0, 5] = 0
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
		E = 1 + np.random.rand()
		nu = 0.49*np.random.rand()
		d = 1 + np.random.rand()
		L = 1 + np.random.rand()
		T = 2*np.random.rand()-1
		rx, ry = np.random.rand(2)
		r = np.array([rx, ry, 0])
		r /= np.linalg.norm(r)
		

		steel = Isotropic(E=E, nu=nu)
		circle = Circle(R=d/2, nu=nu)
		A = (0, 0, 0)
		B = L*r
		bar = EulerBernoulli([A, B])
		bar.material = steel
		bar.section = circle

		U = np.empty((2, 6), dtype="object")
		F = np.zeros((2, 6))

		# U[0, 3:] = 0
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
		E = 1 + np.random.rand()
		nu = 0.49*np.random.rand()
		d = 1 + np.random.rand()
		L = 1 + np.random.rand()
		T = 2*np.random.rand()-1
		ry, rz = np.random.rand(2)
		r = np.array([0, ry, rz])
		r /= np.linalg.norm(r)

		steel = Isotropic(E=E, nu=nu)
		circle = Circle(R=d/2, nu=nu)
		A = (0, 0, 0)
		B = L*r
		bar = EulerBernoulli([A, B])
		bar.material = steel
		bar.section = circle

		U = np.empty((2, 6), dtype="object")
		F = np.zeros((2, 6))

		# U[0, 3:] = 0
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
		E = 1 + np.random.rand()
		nu = 0.49*np.random.rand()
		d = 1 + np.random.rand()
		L = 1 + np.random.rand()
		T = 2*np.random.rand()-1
		rx, rz = np.random.rand(2)
		r = np.array([rx, 0, rz])
		r /= np.linalg.norm(r)

		steel = Isotropic(E=E, nu=nu)
		circle = Circle(R=d/2, nu=nu)
		A = (0, 0, 0)
		B = L*r
		bar = EulerBernoulli([A, B])
		bar.material = steel
		bar.section = circle

		U = np.empty((2, 6), dtype="object")
		F = np.zeros((2, 6))

		# U[0, 3:] = 0
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
		r = np.random.rand(3)  # random direction
		r = np.array([1, 1, 0], dtype="float64")
		r /= np.linalg.norm(r)  # To r a unit vector  
		E = 1 + np.random.rand()
		nu = 0.49*np.random.rand()
		d = 1 + np.random.rand()
		L = 1 + np.random.rand()
		T = 2*np.random.rand()-1

		steel = Isotropic(E=E, nu=nu)
		circle = Circle(R=d/2, nu=nu)
		A = (0, 0, 0)
		B = L*r
		bar = EulerBernoulli([A, B])
		bar.material = steel
		bar.section = circle

		U = np.empty((2, 6), dtype="object")
		F = np.zeros((2, 6))

		# U[0, 3:] = 0
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
@pytest.mark.dependency(depends=["test_tractionR", "test_torsionRcircle"])
def test_end():
	pass

if __name__ == "__main__":
	pytest.main()
	# main()