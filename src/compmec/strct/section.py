import abc
from typing import Optional, Tuple

import numpy as np

from compmec.strct.__classes__ import Material, Profile, Section
from compmec.strct.profile import *


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
    def compute_areas(self) -> Tuple[PositiveFloat]:
        raise NotImplementedError

    @abc.abstractmethod
    def compute_inertias(self) -> Tuple[PositiveFloat]:
        raise NotImplementedError


class RetangularSection(HomogeneousSectionFromMaterialProfile):
    def shear_coefficient(self):
        nu = self.material.nu
        return 10 * (1 + nu) / (12 + 11 * nu)

    def compute_areas(self):
        k = self.shear_coefficient()
        A = np.zeros(3, dtype="float64")
        A[0] = self.profile.area
        A[1] = k * self.profile.area
        A[2] = k * self.profile.area
        return A

    def compute_inertias(self):
        b, h = self.profile.base, self.profile.height
        I = np.zeros(3, dtype="float64")
        I[1] = b * h**3 / 12
        I[2] = h * b**3 / 12
        I[0] = I[1] + I[2]
        print("Warning: Inertia for torsional of retangular is not yet defined")
        return I


class HollowRetangularSection(HomogeneousSectionFromMaterialProfile):
    pass


class CircleSection(HomogeneousSectionFromMaterialProfile):
    def shear_coefficient(self):
        nu = self.material.nu
        return 6 * (1 + nu) / (7 + 6 * nu)

    def compute_areas(self):
        k = self.shear_coefficient()
        A = np.zeros(3, dtype="float64")
        A[0] = self.profile.area
        A[1] = k * self.profile.area
        A[2] = k * self.profile.area
        return A

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

    def compute_areas(self):
        k = self.shear_coefficient()
        A = np.zeros(3, dtype="float64")
        A[0] = self.profile.area
        A[1] = k * self.profile.area
        A[2] = k * self.profile.area
        return A

    def compute_inertias(self):
        Ri4 = self.profile.Ri**4
        Re4 = self.profile.Re**4
        I = np.zeros(3, dtype="float64")
        I[0] = np.pi * (Re4 - Ri4) / 2
        I[1] = np.pi * (Re4 - Ri4) / 4
        I[2] = np.pi * (Re4 - Ri4) / 4
        return I


class PerfilISection(HomogeneousSectionFromMaterialProfile):
    def shear_coefficient(self):
        nu = self.material.nu
        b, h = self.profile.base, self.profile.height
        t1, t2 = self.profile.t1, self.profile.t2
        n = b / h
        m = n * t1 / t2
        pt1 = 12 + 72 * m + 150 * m**2 + 90 * m**3
        pt2 = 11 + 66 * m + 135 * m**2 + 90 * m**3
        pt3 = 10 * n**2 * ((3 + nu) * m + 3 * m**2)
        numerador = 10 * (1 + nu) * (1 + 3 * m) ** 2
        denominador = pt1 + nu * pt2 + pt3
        return numerador / denominador


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
    def A(self, value: Tuple[PositiveFloat]):
        value = np.array(value, dtype="float64")
        if len(value) != 3 or value.ndim != 1:
            raise ValueError("The argument must be 3 floats")
        if np.any(value <= 0):
            raise ValueError("All the elements in value must be positive")
        self._A = value

    @I.setter
    def I(self, value: Tuple[PositiveFloat]):
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
        Retangular: RetangularSection,
        HollowRetangular: HollowRetangularSection,
        Circle: CircleSection,
        HollowCircle: HollowCircleSection,
    }
    for profileclass, sectionclass in mapto.items():
        if type(profile) == profileclass:
            return sectionclass(material, profile)
