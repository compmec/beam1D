
"""
Each point has 6 unknowns:

"""
import numpy as np
from matplotlib import pyplot as plt
from material import Material
from section import Section





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
		
		k1 = E*A/L
		k2 = E*Iz/(L**3)
		k3 = E*Iy/(L**3)
		k4 = G*Ix/L

		K[0, :, 0, :] = np.array([[k1,     0,       0,  0,         0,         0],
								  [ 0,  12*k2,       0,  0,         0,    6*L*k2],
								  [ 0,      0,   12*k3,  0,     -6*k3,         0],
								  [ 0,      0,       0, k4,         0,         0],
								  [ 0,      0, -6*L*k3,  0, 4*L**2*k3,         0],
								  [ 0, 6*L*k2,       0,  0,         0, 4*L**2*k2]])
		K[0, :, 1, :] = np.array([[-k1,       0,       0,   0,         0,          0],
								  [  0,  -12*k2,       0,   0,         0,     6*L*k2],
								  [  0,       0,  -12*k3,   0,     -6*k3,          0],
								  [  0,       0,       0, -k4,         0,          0],
								  [  0,       0,  6*L*k3,   0, 2*L**2*k3,          0],
								  [  0, -6*L*k2,       0,   0,         0, 2*L**2*k2]])
		K[1, :, 0, :] = K[0, :, 1, :].T
		K[1, :, 1, :] = np.array([[k1,       0,       0,  0,         0,          0],
								  [ 0,   12*k2,       0,  0,         0,    -6*L*k2],
								  [ 0,       0,   12*k3,  0,      6*k3,          0],
								  [ 0,       0,       0, k4,         0,          0],
								  [ 0,       0,  6*L*k3,  0, 4*L**2*k3,          0],
								  [ 0, -6*L*k2,       0,  0,         0, 4*L**2*k2]])
		# K = K.reshape((12, 12))
		return K


class Timoshenko(Beam):
	def __init__(self, p0, p1):
		super().__init__(p0, p1)