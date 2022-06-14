
"""
Each point has 6 unknowns:

"""
import numpy as np
from matplotlib import pyplot as plt
from material import Material
from section import Section

class Truss(object):
	def __init__(self, p0, p1):
		self.p0 = np.array(p0)
		self.p1 = np.array(p1)

	@property
	def material(self) -> Material:
		return self._material


	@material.setter
	def material(self, value:Material):
		self._material = value

	@property
	def section(self) -> Section:
		return self._section

	@section.setter
	def section(self, value : Section):
		self._section = value

class Cable(object):
	def __init__(self, p0, p1):
		self.p0 = np.array(p0)
		self.p1 = np.array(p1)

	@property
	def material(self) -> Material:
		return self._material

	
	@material.setter
	def material(self, value:Material):
		self._material = value

	@property
	def section(self) -> Section:
		return self._section

	@section.setter
	def section(self, value : Section):
		self._section = value



class Beam(object):
	def __init__(self, p0, p1):
		self.p0 = np.array(p0)
		self.p1 = np.array(p1)

	@property
	def material(self) -> Material:
		return self._material

	@material.setter
	def material(self, value:Material):
		self._material = value

	@property
	def section(self) -> Section:
		return self._section

	@section.setter
	def section(self, value : Section):
		self._section = value
	

class EulerBernoulli(Beam):
	def __init__(self, p0, p1):
		super().__init__(p0, p1)


	def stiffness_matrix(self):
		"""
		With two points we will have a matrix [12 x 12]
		But we are going to divide the matrix into [x, y, z] coordinates
		That means, our matrix is in fact [4, 3, 4, 3]
		Or also  [2, 6, 2, 6]
		"""
		L = np.sqrt(np.sum((self.p1-self.p0)**2))
		K = np.zeros((2, 6, 2, 6))
		E = self.material.E
		G = self.material.G
		A = self.section.Ax
		Ix = self.section.Ix
		Iy = self.section.Iy
		Iz = self.section.Iz

		Kx = (E*A/L) * (2*np.eye(2)-1)
		Kt = (G*Ix/L) * (2*np.eye(2)-1)
		Ky = (E*Iz/L**3) * np.array([[ 12,    6*L,  -12,    6*L],
									 [6*L, 4*L**2, -6*L, 2*L**2],
									 [-12,   -6*L,   12,   -6*L],
									 [6*L, 2*L**2, -6*L, 4*L**2]])
		Kz = (E*Iy/L**3) * np.array([[  12,   -6*L,  -12,   -6*L],
									 [-6*L, 4*L**2,  6*L, 2*L**2],
									 [ -12,    6*L,   12,    6*L],
									 [-6*L, 2*L**2,  6*L, 4*L**2]])


		K = np.zeros((2, 6, 2, 6))
		for i in range(2):
			for j in range(2):
				K[i, 0, j, 0] = Kx[i, j]
		for i in range(2):
			for j in range(2):
				K[i, 3, j, 3] = Kt[i, j]
		for i in range(2):
			for j in range(2):
				for wa, a in enumerate([1, 5]):
					for wb, b in enumerate([1, 5]):
						K[i, a, j, b] = Ky[2*i+wa, 2*j+wb]
		for i in range(2):
			for j in range(2):
				for wa, a in enumerate([2, 4]):
					for wb, b in enumerate([2, 4]):
						K[i, a, j, b] = Kz[2*i+wa, 2*j+wb]
		# K = K.reshape((12, 12))
		return K


class Timoshenko(Beam):
	def __init__(self, p0, p1):
		super().__init__(p0, p1)



