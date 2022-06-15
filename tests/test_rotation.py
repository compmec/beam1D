import pytest
import numpy as np
import sympy as sp


def random_vector():
	return np.random.rand(3)

def random_unit_vector():
	r = random_vector()
	return r/np.sqrt(np.sum(r*r))


def R(r):
	sign = 1 if r[0] > 0 else -1
	r *= sign
	if sign*r[0] == 1:
		return sign*np.eye(3)
	rx, ry, rz = r
	cos = rx
	sin = np.sqrt(ry**2 + rz**2)
	I = np.eye(3)
	u = (0, rz, -ry)
	ux = np.array([[0, ry, rz],
					[-ry, 0, 0],
					[-rz, 0, 0]])/sin
	uu = np.array([[0, 0, 0],
					[0, rz**2, -ry*rz],
					[0, -ry*rz, ry**2]])/(sin**2)
	Rtot = I * cos + ux * sin + (1-cos)*uu
	if sign < 0:
		return Rtot.T
	return Rtot


def test_randomUnitVector():
	for i in range(10):
		r = random_unit_vector()
		assert np.abs(np.sum(r**2) - 1) < 1e-9

def test_detR():
	for i in range(10):
		r = random_unit_vector()
		Rtot =  R(r)
		det = np.linalg.det(Rtot)
		assert np.abs(det - 1) < 1e-9

def test_invR():
	for i in range(10):
		r = random_unit_vector()
		Rtot = R(r)
		Rinv = np.linalg.inv(Rtot)
		Mat = Rinv @ Rtot
		diff = Mat - np.eye(3)
		assert np.max(diff) < 1e-9

def test_Rtranspose():
	for i in range(10):
		r = random_unit_vector()
		Rtot = R(r)
		Rtra = Rtot.T
		II = Rtra @ Rtot
		diff = np.abs(II - np.eye(3))
		assert np.max(diff) < 1e-9


def test_Rapplytorknown():
	for rx in (-1, 0, 1):
		for ry in (-1, 0, 1):
			for rz in (-1, 0, 1):
				abso = np.sqrt(rx**2+ry**2+rz**2)
				if abso == 0:
					continue
				r = np.array([rx, ry, rz])/abso
				Rtot = R(r)
				rt = Rtot @ r
				diff = rt - np.array([1, 0, 0])
				diff = np.abs(diff)
				assert np.max(diff) < 1e-9


def test_Rapplytor():
	for i in range(10):
		r = random_unit_vector()
		Rtot = R(r)
		rt = Rtot @ r
		diff = rt - np.array([1, 0, 0])
		diff = np.abs(diff)
		assert np.max(diff) < 1e-9


def test_RminusR():
	for i in range(10):
		r = random_unit_vector()
		Rtot = R(r)
		Rles = R(-r)
		diff = np.abs(Rles.T - Rtot)
		assert np.max(diff) < 1e-9




	
	


def main():
	rx, ry, rz = sp.symbols("rx ry rz")
	cos = rx
	sin = sp.sqrt(ry**2 + rz**2)
	I = sp.eye(3)
	print(type(I))
	u = (0, rz, -ry)
	ux = sp.Matrix([[0, ry, rz],
					[-ry, 0, 0],
					[-rz, 0, 0]])
	uu = sp.Matrix([[0, 0, 0],
					[0, rz**2, -ry*rz],
					[0, -ry*rz, ry**2]])
	R = I * cos + ux + (1-rx) * uu / (ry**2+rz**2)
	# R /= sp.sqrt(rx**2+ry**2+rz**2)
	R = sp.expand(R)
	R = sp.simplify(R)

	print("R = ")
	print(R)
	det = R.det()
	det = sp.expand(det)
	det = sp.simplify(det)
	print("det = ")
	print(det)


if __name__ == "__main__":
	pytest.main()
	# main()