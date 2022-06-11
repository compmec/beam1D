
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


class Timoshenko(Beam):
	def __init__(self, p0, p1):
		super().__init__(p0, p1)