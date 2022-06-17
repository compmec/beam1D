import numpy as np

class Material(object):
	def __init__(self):
		super().__init__(self)

class Isotropic(Material):
	def __doc__(self):
		"""
		Receives the configuration material.
		It's isotropic
		"""
		pass

	def __init__(self, *args, **kwargs):
		self.init_variables()

		if "E" in kwargs:
			self.E = kwargs["E"]
		if "G" in kwargs:
			self.G = kwargs["G"]
		if "K" in kwargs:
			self.K = kwargs["K"]
		if "nu" in kwargs: 
			self.nu = kwargs["nu"]
		# if "lambda" in kwargs:
		# 	self.Lame1 = kwargs["lambda"]
		# if "mu" in kwargs:
		# 	self.G = kwargs["mu"]
		if "Lame1" in kwargs:
			self.Lame1 = kwargs["Lame1"]
		if "Lame2" in kwargs:
			self.G = kwargs["Lame2"]
		self.__compute_all()

	def init_variables(self):
		self._E = None
		self._G = None
		self._K = None
		self._nu = None
		self._lambda = None

	def __cannot_compute(self, var):
		msg = "Cannot compute '%s'. Current variables = \n" % var
		msg += "       E = %s\n" % self.E
		msg += "       G = %s\n" % self.G
		msg += "       K = %s\n" % self.K
		msg += "      nu = %s\n" % self.nu
		msg += "   Lame1 = %s\n" % self.Lame1
		msg += "   Lame2 = %s\n" % self.Lame2 

		raise Exception(msg)
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
		return self._E

	@property
	def G(self) -> float:
		return self._G

	@property
	def K(self) -> float:
		return self._K

	@property
	def nu(self) -> float:
		return self._nu

	@property
	def Lame1(self) -> float:
		return self._lambda

	@property
	def Lame2(self) -> float:
		return self._G

	@E.setter
	def E(self, value : float):
		self._E = value

	@G.setter
	def G(self, value : float):
		self._G = value
	
	@K.setter
	def K(self, value : float):
		self._K = value

	@nu.setter
	def nu(self, value : float):
		if 0.49 < value and value < 0.5:
			raise Exception("Poisson is near 0.5. We cannot treat non-compressible materials")
		if value < 0 or 0.49 < value:
			raise Exception("Poisson modulus must be between [0, 0.49]")
		
		self._nu = value

	@Lame1.setter
	def Lame1(self, value : float):
		self._lambda = value

	@Lame2.setter
	def Lame2(self, value : float):
		self.G = value

	
