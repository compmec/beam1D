import abc
from typing import Optional, Tuple

import numpy as np

from compmec.strct import profile as prof
from compmec.strct.__classes__ import Material, Profile, Section


class HomogeneousSection(Section):
    @property
    def A(self) -> Tuple[float, float, float]:
        return tuple(self._A)

    @property
    def I(self) -> Tuple[float, float, float]:
        return tuple(self._I)


class HomogeneousSectionFromMaterialProfile(HomogeneousSection):
    def __init__(self, material: Material, profile: Profile):
        if not isinstance(material, Material):
            raise TypeError
        if not isinstance(profile, Profile):
            raise TypeError
        self._material = material
        self._profile = profile
        self._A = self.compute_areas()
        self._I = self.compute_inertias()

    @property
    def material(self) -> Material:
        return self._material

    @property
    def profile(self) -> Profile:
        return self._profile

    @abc.abstractmethod
    def shear_coefficient(self) -> float:
        raise NotImplementedError

    def compute_areas(self):
        k = self.shear_coefficient()
        A = np.zeros(3, dtype="float64")
        A[0] = self.profile.area
        A[1] = k * self.profile.area
        A[2] = k * self.profile.area
        return A

    @abc.abstractmethod
    def compute_inertias(self) -> Tuple[float]:
        raise NotImplementedError


class RetangularSection(HomogeneousSectionFromMaterialProfile):
    def shear_coefficient(self):
        nu = self.material.nu
        return 10 * (1 + nu) / (12 + 11 * nu)

    def compute_inertias(self):
        b, h = self.profile.base, self.profile.height
        I = np.zeros(3, dtype="float64")
        I[1] = b * h**3 / 12
        I[2] = h * b**3 / 12
        I[0] = I[1] + I[2]
        print("Warning: Inertia for torsional of retangular is not yet defined")
        return I


class HollowRetangularSection(HomogeneousSectionFromMaterialProfile):
    def shear_coefficient(self) -> float:
        nu = self.material.nu
        bi, be = self.profile.bi, self.profile.be
        hi, he = self.profile.hi, self.profile.he
        t1, t2 = 0.5 * (he - hi), 0.5 * (be - bi)
        n = bi / hi
        m = n * t1 / t2
        numer = 10 * (1 + nu) * (1 + 3 * m) ** 2
        deno1 = 12 + 72 * m + 150 * m**2 + 90 * m**3
        deno2 = 11 + 66 * m + 135 * m**2 + 90 * m**3
        deno3 = 10 * n**2 * m * (3 + nu + 3 * m)
        return numer / (deno1 + nu * deno2 + deno3)

    def compute_inertias(self):
        bi, be = self.profile.bi, self.profile.be
        hi, he = self.profile.hi, self.profile.he
        I = np.zeros(3, dtype="float64")
        I[1] = (be * he**3 - bi * hi**3) / 12
        I[2] = (he * be**3 - hi * bi**3) / 12
        I[0] = I[1] + I[2]
        print("Warning: Inertia for torsional of retangular is not yet defined")
        return I


class CircleSection(HomogeneousSectionFromMaterialProfile):
    def shear_coefficient(self) -> float:
        nu = self.material.nu
        return 6 * (1 + nu) / (7 + 6 * nu)

    def compute_inertias(self):
        R4 = self.profile.radius**4
        I = np.zeros(3, dtype="float64")
        I[0] = np.pi * R4 / 2
        I[1] = np.pi * R4 / 4
        I[2] = np.pi * R4 / 4
        return I


class HollowCircleSection(HomogeneousSectionFromMaterialProfile):
    def shear_coefficient(self):
        Ri, Re = self.profile.Ri, self.profile.Re
        nu = self.material.nu
        m2 = (Ri / Re) ** 2
        return 6 * (1 + nu) / ((7 + 6 * nu) + 4 * m2 * (5 + 3 * nu) / (1 + m2) ** 2)

    def compute_inertias(self):
        Ri4 = self.profile.Ri**4
        Re4 = self.profile.Re**4
        inertia = np.zeros(3, dtype="float64")
        inertia[0] = np.pi * (Re4 - Ri4) / 2
        inertia[1] = np.pi * (Re4 - Ri4) / 4
        inertia[2] = np.pi * (Re4 - Ri4) / 4
        return inertia


class PerfilISection(HomogeneousSectionFromMaterialProfile):
    def shear_coefficient(self):
        nu = self.material.nu
        b, h = self.profile.b, self.profile.h
        t, s = self.profile.t, self.profile.s
        n = b / h
        m = n * t / s
        pt1 = 12 + 72 * m + 150 * m**2 + 90 * m**3
        pt2 = 11 + 66 * m + 135 * m**2 + 90 * m**3
        pt3 = 10 * n**2 * ((3 + nu) * m + 3 * m**2)
        numerador = 10 * (1 + nu) * (1 + 3 * m) ** 2
        denominador = pt1 + nu * pt2 + pt3
        return numerador / denominador

    def compute_inertias(self):
        b, h = self.profile.b, self.profile.h
        t, s = self.profile.t, self.profile.s
        Iz = (b * t / 6) * (t**2 + 3 * h**2) + s * (h - t) ** 3 / 12  # int z^2
        Iy = t * b**3 / 6 + (h - t) * s**3 / 12  # int y^2
        inertia = np.zeros(3, dtype="float64")
        inertia[0] = Iy + Iz
        inertia[1] = Iy
        inertia[2] = Iz
        return inertia


class GeneralSection(HomogeneousSection):
    def __init__(self):
        self._A = None
        self._I = None

    @property
    def A(self) -> Tuple[float, float, float]:
        if self._A is None:
            raise ValueError("You must set A to use general section!")
        return tuple(self._A)

    @property
    def I(self) -> Tuple[float, float, float]:
        if self._I is None:
            raise ValueError("You must set I to use general section!")
        return tuple(self._I)

    @A.setter
    def A(self, value: Tuple[float]):
        value = np.array(value, dtype="float64")
        if len(value) != 3 or value.ndim != 1:
            raise ValueError("The argument must be 3 floats")
        if np.any(value <= 0):
            raise ValueError("All the elements in value must be positive")
        self._A = value

    @I.setter
    def I(self, value: Tuple[float]):
        value = np.array(value, dtype="float64")
        if len(value) != 3 or value.ndim != 1:
            raise ValueError("The argument must be 3 floats")
        if np.any(value <= 0):
            raise ValueError("All the elements in value must be positive")
        self._I = value


def create_section_from_material_profile(
    material: Material, profile: Profile
) -> HomogeneousSection:
    if not isinstance(material, Material):
        raise TypeError
    if not isinstance(profile, Profile):
        raise TypeError
    mapto = {
        prof.Retangular: RetangularSection,
        prof.HollowRetangular: HollowRetangularSection,
        prof.Circle: CircleSection,
        prof.HollowCircle: HollowCircleSection,
    }
    for profileclass, sectionclass in mapto.items():
        if type(profile) == profileclass:
            return sectionclass(material, profile)
