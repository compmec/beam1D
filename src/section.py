import numpy as np

class Section(object):
	def __init__(self):
		self._A = np.zeros(3, dtype="float64")
		self._I = np.zeros(3, dtype="float64")

	@property
	def Ax(self) -> float:
		return self._A[0]

	@property
	def Ay(self) -> float:
		return self._A[1]

	@property
	def Az(self) -> float:
		return self._Az[2]

	@property
	def A(self) -> np.ndarray:
		return self._A
	
	
	@property
	def Ix(self) -> float:
		return self._I[0]

	@property
	def Iy(self) -> float:
		return self._I[1]

	@property
	def Iz(self) -> float:
		return self._I[2]

	@property
	def I(self) -> np.ndarray:
		return self._I


	@Ax.setter
	def Ax(self, value:float):
		self._A[0] = value

	@Ay.setter
	def Ay(self, value:float):
		self._A[1] = value

	@Az.setter
	def Az(self, value:float):
		self._A[2] = value

	@Ix.setter
	def Ix(self, value:float):
		self._I[0] = value

	@Iy.setter
	def Iy(self, value:float):
		self._I[1] = value

	@Iz.setter
	def Iz(self, value:float):
		self._I[2] = value


class Retangular(Section):
	def __doc__(self):
		"""
		Returns a square section
		"""
	def __init__(self, b: float, h:float, nu:float):
		"""
		b is the lower base
		h is the height
		"""
		self.Ax = a*b
		k = 10*(1+nu)/(12+11*nu)
		self.Ay = k * self.Ax
		self.Az = k * self.Ax
		self.Ix = b*h*(b**2 + h**2)/12  # Wrong, not finished
		self.Iy = b*h**3/12
		self.Iz = b**3 * h/12

class ThinHollowSquare(Section):
	def __doc__(self):
		"""
		Docs
		"""
	def __init__(self, side:float, nu:float):

		k = 20*(1+nu)/(4+3*nu)

class Circle(Section):
	def __init__(self, R:float, nu:float):
		super().__init__()
		self.Ax = np.pi* R**2
		k = 6*(1+nu)/(7+6*nu)
		self.Ay = k * self.Ax
		self.Az = k * self.Ax
		d = 2 * R
		self.Ix = np.pi * d**4 / 32
		self.Iy = np.pi * d**4 / 64
		self.Iz = np.pi * d**4 / 64 


class HollowCircle(Section):
	def __init__(self, Ri:float, Re:float, nu:float):
		super().__init__()
		m = Ri/Re
		k = 6*(1+nu)*((1+m**2)**2)/( (7+6*nu)*(1+m**2)**2 + (20+12*nu)*m**2)

		self.Ax = np.pi* (Re**2-Ri**2)
		k = 6*(1+nu)/(7+6*nu)
		self.Ay = k * self.Ax
		self.Az = k * self.Ax
		di = 2 * Ri
		de = 2 * Re
		self.Ix = np.pi * (de**4-di**4) / 32
		self.Iy = np.pi * (de**4-di**4) / 64
		self.Iz = np.pi * (de**4-di**4) / 64 

class ThinCircle(Section):
	"""
	We suppose that the thickness is 0.01 * R
	"""
	def __init__(self, R:float, nu:float):
		super().__init__()
		e = 0.01 * R
		k = 2*(1+nu)/(4+3*nu)
		self.Ax = 2 * np.pi * R * e
		self.Ay = k * self.Ax
		self.Az = k * self.Ax
		self.Ix = 2 * np.pi*e * R**3
		self.Iy = np.pi * e * R**3
		self.Iz = np.pi * e * R**3

class PerfilI(Section):
	def __init__(self, b:float, h:float, t1:float, t2:float):
		super().__init__()
		m = b*t1/(h*t2)
		n = b/h
		pt1 = 12+72*m + 150*m**2 + 90*m**3
		pt2 = 11+66*m + 135*m**2 + 90*m**3
		pt3 = 10*n**2 * ((3+nu)*m + 3*m**2)
		self.k = 10*(1+nu)*(1+3*m)**2/(pt1 + nu * pt2 + pt3)

class General(Section):
	def __init__(self, curves: list):
		"""
		curves is a list of closed curves that defines the geometry
		Each curve is a Nurbs, with the points.
		It's possible to have a circle, only with one curve, a circle
		Until now, it's not implemented
		"""
		super().__init__()
		raise Exception("Not implemented")
	


def main():
	pass

if __name__ == "__main__":
	main()
