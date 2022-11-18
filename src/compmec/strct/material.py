import numpy as np
from compmec.strct.__classes__ import Material


class Isotropic(Material):
	def __doc__(self):
		"""
		Receives the configuration material.
		It's isotropic
		"""
		pass

	@staticmethod
	def isFloat(value: float):
		try:
			float(value)
		except Exception as e:
			raise TypeError(f"The received value has type {type(value)}, must be float")

	@staticmethod
	def isPositive(value: float):
		if value <= 0:
			raise ValueError(f"The value must be positive! Received {value}")

	def __init__(self, *args, **kwargs):
		self.__init_variables()
		self.__fill_variables(**kwargs)
		self.__compute_all()

	def __init_variables(self):
		self.__E = None
		self.__G = None
		self.__K = None
		self.__nu = None
		self.__lambda = None

	def __fill_variables(self, **kwargs):
		if "E" in kwargs:
			self.E = kwargs["E"]
		if "G" in kwargs:
			self.G = kwargs["G"]
		if "K" in kwargs:
			self.K = kwargs["K"]
		if "nu" in kwargs: 
			self.nu = kwargs["nu"]
		if "Lame1" in kwargs:
			self.Lame1 = kwargs["Lame1"]  # lambda
		if "Lame2" in kwargs:
			self.Lame2 = kwargs["Lame2"]  # mu

	def __cannot_compute(self, var):
		msg = "Cannot compute '%s'. Current variables = \n" % var
		msg += "       E = %s\n" % self.E
		msg += "       G = %s\n" % self.G
		msg += "       K = %s\n" % self.K
		msg += "      nu = %s\n" % self.nu
		msg += "   Lame1 = %s\n" % self.Lame1
		msg += "   Lame2 = %s\n" % self.Lame2 
		raise ValueError(msg)
		
	def __compute_E(self):
		if self.E is not None:
			return
		if self.K is not None:
			K = self.K
			if self.Lame1 is not None:
				L = self.Lame1
				self.E = 9*K*(K-L)/(3*K-L)
			elif self.G is not None:
				G = self.G
				self.E = 9*K*G/(3*K+G)
			elif self.nu is not None:
				self.E = 3*K*(1-2*self.nu)
			else:
				self.__cannot_compute("E")
		elif self.Lame1 is not None:
			L = self.Lame1
			if self.G is not None:
				G = self.G
				self.E = G*(3*L+2*G)/(L+G)
			elif self.nu is not None:
				nu = self.nu
				self.E = L*(1+nu)*(1-2*nu)/nu
			else:
				self.__cannot_compute("E")
		elif self.G is not None:
			G = self.G
			if self.nu is not None:
				self.E = 2*G*(1+self.nu)
			else:
				self.__cannot_compute("E")
		else:
			self.__cannot_compute("E")

	def __compute_K(self):
		if self.K is not None:
			return
		if self.E is not None:
			if self.G is not None:
				self.K = self.E*self.G/(3*(3*self.G-self.E))
			elif self.nu is not None:
				self.K = self.E/(3*(1-2*self.nu))
			else:
				self.__cannot_compute("K")
		else:
			self.__cannot_compute("K")



	def __compute_G(self):
		if self.G is not None:
			return
		if self.E is not None:
			if self.nu is not None:
				self.G = self.E/(2*(1+self.nu)) 
			elif self.K is not None:
				self.G = 3*self.K*self.E/(9*self.K - self.E)
			elif self.Lame1 is not None:
				R = np.sqrt(self.E**2 + 9*self.Lame1**2 + 2*self.E*self.Lame1)
				self.G = (self.E-3*self.Lame1+R)/4
			else:
				self.__cannot_compute("G")
		else:
			self.__cannot_compute("G")

	def __compute_nu(self):
		if self.nu is not None:
			return
		if self.E is not None:
			if self.G is not None:
				self.nu = self.E/(2*self.G) - 1 
			else:
				self.__cannot_compute("nu")
		else:
			self.__cannot_compute("nu")

	def __compute_Lame1(self):
		if self.Lame1 is not None:
			return
		if self.G is not None:
			if self.nu is not None:
				self.Lame1 = 2*self.G*self.nu/(1-2*self.nu)
			else:
				self.__cannot_compute("Lame1")
		else:
			self.__cannot_compute("Lame1")

	def __compute_all(self):
		self.__compute_E()
		self.__compute_G()
		self.__compute_K()
		self.__compute_nu()
		self.__compute_Lame1()

	@property
	def E(self) -> float:
		return self.__E

	@property
	def G(self) -> float:
		return self.__G

	@property
	def K(self) -> float:
		return self.__K

	@property
	def nu(self) -> float:
		return self.__nu

	@property
	def Lame1(self) -> float:
		return self.__lambda

	@property
	def Lame2(self) -> float:
		return self.__G

	@E.setter
	def E(self, value : float):
		Isotropic.isFloat(value)
		Isotropic.isPositive(value)
		self.__E = float(value)

	@G.setter
	def G(self, value : float):
		Isotropic.isFloat(value)
		Isotropic.isPositive(value)
		self.__G = float(value)
	
	@K.setter
	def K(self, value : float):
		Isotropic.isFloat(value)
		Isotropic.isPositive(value)
		self.__K = float(value)

	@nu.setter
	def nu(self, value : float):
		Isotropic.isFloat(value)
		Isotropic.isPositive(value)
		if 0.49 < value and value < 0.5:
			raise ValueError("Poisson is near 0.5. We cannot treat non-compressible materials")
		if value < 0 or 0.49 < value:
			raise ValueError("Poisson modulus must be between [0, 0.49]")
		
		self.__nu = value

	@Lame1.setter
	def Lame1(self, value : float):
		Isotropic.isFloat(value)
		Isotropic.isPositive(value)
		self.__lambda = value

	@Lame2.setter
	def Lame2(self, value : float):
		self.G = value