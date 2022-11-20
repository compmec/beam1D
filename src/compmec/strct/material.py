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

    def __init__(self, **kwargs):
        self.__init_variables()
        self.__valid_kwargs(kwargs)
        self.__fill_variables(**kwargs)

    def __init_variables(self):
        self.__E = None
        self.__G = None
        self.__K = None
        self.__nu = None
        self.__lambda = None

    def __valid_kwargs(**kwargs):
        for key, item in kwargs.items():
            try:
                float(item)
            except Exception as e:
                raise TypeError(f"All the elements in the vector must be floats: kwargs[{key}] = {type(item)}")
            if float(item) < 0:
                raise ValueError("All the elements must be positive!")

    def __fill_variables(self, **kwargs):
        if len(kwargs) != 2:
            raise ValueError(f"The numbers of variables must be two! Received {len(kwargs)}")
        if "Lame1" in kwargs:
            kwargs["L"] = kwargs["Lame1"]
        if "Lame2" in kwargs:
            kwargs["G"] = kwargs["Lame2"]
        if "E" in kwargs and "nu" in kwargs:
            self.__compute_from_Enu(kwargs["E"], kwargs["nu"])
        elif "E" in kwargs and "G" in kwargs:
            self.__compute_from_EG(kwargs["E"], kwargs["G"])
        elif "L" in kwargs and "G" in kwargs:
            self.__compute_from_LG(kwargs["L"], kwargs["G"])
        elif "L" in kwargs and "E" in kwargs:
            self.__compute_from_LE(kwargs["L"], kwargs["E"])
        elif "K" in kwargs and "L" in kwargs:
            self.__compute_from_KL(kwargs["K"], kwargs["L"])
        elif "K" in kwargs and "G" in kwargs:
            self.__compute_from_KG(kwargs["K"], kwargs["G"])
        elif "L" in kwargs and "nu" in kwargs:
            self.__compute_from_Lnu(kwargs["L"], kwargs["nu"])
        elif "G" in kwargs and "nu" in kwargs:
            self.__compute_from_Gnu(kwargs["G"], kwargs["nu"])
        elif "K" in kwargs and "nu" in kwargs:
            self.__compute_from_Knu(kwargs["K"], kwargs["nu"])
        elif "K" in kwargs and "E" in kwargs:
            self.__compute_from_KE(kwargs["K"], kwargs["E"])
        else:
            raise ValueError(f"Cannot compute with the arguments {kwargs.keys()}")
        

    def __compute_from_EG(self, E: float, G: float):
        self.E = E
        self.G = G
        self.nu = E/(2*G) - 1
        self.K = E*G/(3*(3*G-E))
        self.Lame1 = G*(E-2*G)/(3*G-E)

    def __compute_from_LE(self, L: float, E: float):
        """
        Wikipedia website don't give a direct relation.
        We use that E/L = (1+nu)*(1-2*nu)/nu
        """
        self.Lame1 = L
        self.E = E
        r = E/L
        self.nu = (np.sqrt(9+2*r+r**2)-(1+r))/4
        self.G = E/(2*(1+self.nu))
        self.K = L + 2*self.G/3

    def __compute_from_LG(self, L: float, G: float):
        self.Lame1 = L
        self.G = G
        self.E = G*(3*L+2*G)/(L+G)
        self.K = L + 2*G/3
        self.nu = L/(2*(L+G))

    def __compute_from_KL(self, K: float, L: float):
        self.K = K
        self.Lame1 = L
        self.E = 9*K*(K-L)/(3*K-L)
        self.G = 3*(K-L)/2
        self.nu = L/(3*K-L)

    def __compute_from_KG(self, K: float, G: float):
        self.K = K
        self.G = G
        self.Lame1 = K - 2*G/3
        self.E = 9*K*G/(3*K+G)
        self.nu = (3*K-2*G)/(2*(3*K+G))
    
    def __compute_from_Lnu(self, L: float, nu: float):
        self.Lame1 = L
        self.nu = nu
        self.K = L*(1+nu)/(3*nu)
        self.E = L*(1+nu)*(1-2*nu)/nu
        self.G = L*(1-2*nu)/(2*nu)

    def __compute_from_Gnu(self, G: float, nu: float):
        self.nu = nu
        self.G = G
        self.Lame1 = 2*G*nu/(1-2*nu)
        self.E = 2*G*(1+nu)
        self.K = 2*G*(1+nu)/(3*(1-2*nu))

    def __compute_from_Enu(self, E: float, nu: float):
        self.E = E
        self.nu = nu
        self.K = E/(3*(1-2*nu))
        self.G = E/(2*(1+nu)) 
        self.Lame1 = 2*self.G*nu/(1-2*nu)


    def __compute_from_Knu(self, K: float, nu: float):
        self.K = K
        self.nu = nu
        self.E = 3*K*(1-2*nu)
        self.G = 3*K*(1-2*nu)/(2*(1+nu))
        self.Lame1 = 3*K*nu/(1+nu)

    def __compute_from_KE(self, K: float, E: float):
        self.K = K
        self.E = E
        self.Lame1 = 3*K*(3*K-E)/(9*K-E)
        self.G = 3*K*E/(9*K-E)
        self.nu = (3*K-E)/(6*K)
     
    

    


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