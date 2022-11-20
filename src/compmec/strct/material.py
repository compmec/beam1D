import numpy as np

from compmec.strct.__classes__ import Material
from compmec.strct.verifytype import PositiveFloat


class Isotropic(Material):
    def __init__(self, **kwargs):
        self.__init_variables()
        self.__valid_kwargs(kwargs)
        self.__fill_variables(**kwargs)
        self.__verify_all_final()

    def __init_variables(self):
        self.__E = None
        self.__G = None
        self.__K = None
        self.__nu = None
        self.__lambda = None

    def __valid_kwargs(self, kwargs):
        for key, item in kwargs.items():
            PositiveFloat.verify(item, key)

    def __fill_variables(self, **kwargs):
        if len(kwargs) != 2:
            raise ValueError(
                f"The numbers of variables must be two! Received {len(kwargs)}"
            )
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

    def __verify_all_final(self):
        PositiveFloat.verify(self.E, "E")
        PositiveFloat.verify(self.G, "G")
        PositiveFloat.verify(self.K, "K")
        PositiveFloat.verify(self.nu, "nu")
        PositiveFloat.verify(self.Lame1, "Lame1")
        if 0.49 < self.nu and self.nu < 0.5:
            raise ValueError(
                "Poisson is near 0.5. We cannot treat non-compressible materials"
            )

    def __compute_from_EG(self, E: float, G: float):
        self.__E = E
        self.__G = G
        self.__nu = E / (2 * G) - 1
        self.__K = E * G / (3 * (3 * G - E))
        self.__L = G * (E - 2 * G) / (3 * G - E)

    def __compute_from_LE(self, L: PositiveFloat, E: PositiveFloat):
        """
        Wikipedia website don't give a direct relation.
        We use that E/L = (1+nu)*(1-2*nu)/nu
        """
        self.__L = L
        self.__E = E
        r = E / L
        self.__nu = (np.sqrt(9 + 2 * r + r**2) - (1 + r)) / 4
        self.__G = E / (2 * (1 + self.nu))
        self.__K = L + 2 * self.G / 3

    def __compute_from_LG(self, L: PositiveFloat, G: PositiveFloat):
        self.__L = L
        self.__G = G
        self.__E = G * (3 * L + 2 * G) / (L + G)
        self.__K = L + 2 * G / 3
        self.__nu = L / (2 * (L + G))

    def __compute_from_KL(self, K: PositiveFloat, L: PositiveFloat):
        self.__K = K
        self.__L = L
        self.__E = 9 * K * (K - L) / (3 * K - L)
        self.__G = 3 * (K - L) / 2
        self.__nu = L / (3 * K - L)

    def __compute_from_KG(self, K: PositiveFloat, G: PositiveFloat):
        self.__K = K
        self.__G = G
        self.__L = K - 2 * G / 3
        self.__E = 9 * K * G / (3 * K + G)
        self.__nu = (3 * K - 2 * G) / (2 * (3 * K + G))

    def __compute_from_Lnu(self, L: PositiveFloat, nu: PositiveFloat):
        self.__L = L
        self.__nu = nu
        self.__K = L * (1 + nu) / (3 * nu)
        self.__E = L * (1 + nu) * (1 - 2 * nu) / nu
        self.__G = L * (1 - 2 * nu) / (2 * nu)

    def __compute_from_Gnu(self, G: PositiveFloat, nu: PositiveFloat):
        self.__nu = nu
        self.__G = G
        self.__L = 2 * G * nu / (1 - 2 * nu)
        self.__E = 2 * G * (1 + nu)
        self.__K = 2 * G * (1 + nu) / (3 * (1 - 2 * nu))

    def __compute_from_Enu(self, E: PositiveFloat, nu: PositiveFloat):
        self.__E = E
        self.__nu = nu
        self.__K = E / (3 * (1 - 2 * nu))
        self.__G = E / (2 * (1 + nu))
        self.__L = 2 * self.G * nu / (1 - 2 * nu)

    def __compute_from_Knu(self, K: PositiveFloat, nu: PositiveFloat):
        self.__K = K
        self.__nu = nu
        self.__E = 3 * K * (1 - 2 * nu)
        self.__G = 3 * K * (1 - 2 * nu) / (2 * (1 + nu))
        self.__L = 3 * K * nu / (1 + nu)

    def __compute_from_KE(self, K: PositiveFloat, E: PositiveFloat):
        self.__K = K
        self.__E = E
        self.__L = 3 * K * (3 * K - E) / (9 * K - E)
        self.__G = 3 * K * E / (9 * K - E)
        self.__nu = (3 * K - E) / (6 * K)

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
        return self.__L

    @property
    def Lame2(self) -> float:
        return self.__G
