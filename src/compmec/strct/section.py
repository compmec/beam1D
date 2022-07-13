import numpy as np
import pygmsh
from compmec.strct.__classes__ import Section



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
        super().__init__()
        self.nu = nu
        self.b = b
        self.h = h
        self.__compute_areas()
        self.__compute_inertias()

    def __compute_areas(self):
        k = 10*(1+self.nu)/(12+11*self.nu)
        self.Ax = self.b*self.h
        self.Ay = k * self.Ax
        self.Az = k * self.Ax

    def __compute_inertias(self):
        self.Iy = self.b*self.h**3/12
        self.Iz = self.h*self.b**3/12
        self.Ix = self.Iy + self.Iz
        print("Warning: Inertia for torsional of retangular is not yet defined")
        # raise NotImplementedError("Torsion for a retangular is not defined yet")

class HollowRetangular(Section):
    def __init__(self, be: float, he:float, bi:float, hi:float, nu:float):
        """
        b is the lower base
        h is the height
        """
        super().__init__()
        self.nu = nu
        self.be = be
        self.he = he
        self.bi = bi
        self.hi = hi
        self.__compute_areas()
        self.__compute_inertias()

    def __compute_areas(self):
        raise NotImplementedError("Areas for hollow retangular still not defined")

    def __compute_inertias(self):
        raise NotImplementedError("Inertias for hollow retangular still not defined")

class ThinRetangular(Section):
    def __init__(self, b: float, h:float, nu:float):
        """
        b is the lower base
        h is the height
        """
        super().__init__()
        self.nu = nu
        self.b = b
        self.h = h
        self.__compute_areas()
        self.__compute_inertias()

    def __compute_areas(self):
        raise NotImplementedError("Areas for thin retangular still not defined")

    def __compute_inertias(self):
        raise NotImplementedError("Inertias for thin retangular still not defined")

class Square(Section):
    def __doc__(self):
        """
        Docs
        """
    def __init__(self, b:float, nu:float):
        super().__init__()
        self.nu = nu
        self.b = b
        self.__compute_areas()
        self.__compute_inertias()

    def __compute_areas(self):
        k = 20*(1+self.nu)/(4+3*self.nu)
        self.Ax = self.b**2
        self.Ay = k * self.Ax
        self.Az = k * self.Ax
        print("Warning: Areas for a square is not yet well defined")
        # raise NotImplementedError("Areas for a square are not defined")

    def __compute_inertias(self):
        print("Warning: Inertias for a square is not yet well defined")
        self.Iy = self.b**4/12
        self.Iz = self.Iy
        self.Ix = 2*self.Iy
        # raise NotImplementedError("Inertias for a square are not defined")

class HollowSquare(Section):
    def __doc__(self):
        """
        Docs
        """
    def __init__(self, be:float, bi:float, nu:float):
        super().__init__()
        self.nu = nu
        self.be = be
        self.bi = bi
        self.__compute_areas()
        self.__compute_inertias()

    def __compute_areas(self):
        raise NotImplementedError("Areas for a hollow square are not defined")

    def __compute_inertias(self):
        raise NotImplementedError("Inertias for a hollow square are not defined")

class ThinSquare(Section):
    def __doc__(self):
        """
        Docs
        """
    def __init__(self, b:float, nu:float):
        super().__init__()
        self.nu = nu
        self.b = b
        self.__compute_areas()
        self.__compute_inertias()

    def __compute_areas(self):
        raise NotImplementedError("Areas for a hollow square are not defined")

    def __compute_inertias(self):
        raise NotImplementedError("Inertias for a Thin square are not defined")

class Circle(Section):
    def __init__(self, R:float, nu:float):
        super().__init__()
        self.nu = nu
        self.R = R
        self.__compute_areas()
        self.__compute_inertias()

    def __compute_areas(self):
        k = 6*(1+self.nu)/(7+6*self.nu)
        self.Ax = np.pi* self.R**2
        self.Ay = k * self.Ax
        self.Az = k * self.Ax
    
    def __compute_inertias(self):
        R4 = self.R**4
        self.Ix = np.pi * R4 / 2
        self.Iy = np.pi * R4 / 4
        self.Iz = np.pi * R4 / 4
    
    def triangular_mesh(self, meshsize:float):
        with pygmsh.geo.Geometry() as geom:
            geom.add_circle((0, 0), self.R, mesh_size=meshsize)
            mesh = geom.generate_mesh()
        return mesh


class HollowCircle(Section):
    def __init__(self, Ri:float, Re:float, nu:float):
        super().__init__()
        self.nu = nu
        self.Ri = Ri
        self.Re = Re
        self.__compute_areas()
        self.__compute_inertias()

    def __compute_areas(self):
        m = self.Ri/self.Re
        nu = self.nu
        k = 6*(1+nu)*((1+m**2)**2)/( (7+6*nu)*(1+m**2)**2 + (20+12*nu)*m**2)
        self.Ax = np.pi* self.Re**2 *(1-m**2)
        self.Ay = k * self.Ax
        self.Az = k * self.Ax

    def __compute_inertias(self):
        Ri4 = self.Ri**4
        Re4 = self.Re**4
        self.Ix = np.pi * (Re4 - Ri4) / 2
        self.Iy = np.pi * (Re4 - Ri4) / 4
        self.Iz = np.pi * (Re4 - Ri4) / 4

    def triangular_mesh(self, meshsize: float):
        with pygmsh.occ.Geometry() as geom:
            geom.characteristic_length_max = meshsize
            externalcircle = geom.add_disk((0, 0), self.Re)
            internalcircle = geom.add_disk((0, 0), self.Ri) 
            geom.boolean_difference(externalcircle, internalcircle)
            mesh = geom.generate_mesh()
        return mesh

class ThinCircle(Section):
    """
    We suppose that the thickness is 0.01 * R
    """
    def __init__(self, R:float, nu:float):
        super().__init__()
        self.nu = nu
        self.R = R
        self.e = 0.01*self.R
        self.__compute_areas()
        self.__compute_inertias

    def __compute_areas(self):
        k = 2*(1+self.nu)/(4+3*self.nu)
        self.Ax = 2 * np.pi * self.R * self.e
        self.Ay = k * self.Ax
        self.Az = k * self.Ax
    
    def __compute_inertias(self):
        eR3 = self.e * self.R**3
        self.Ix = 2 * np.pi * eR3
        self.Iy = np.pi * eR3
        self.Iz = np.pi * eR3

    def triangular_mesh(self):
        with pygmsh.occ.Geometry() as geom:
            geom.characteristic_length_max = self.e
            externalcircle = geom.add_disk((0, 0), self.R + 0.5*self.e)
            internalcircle = geom.add_disk((0, 0), self.R - 0.5*self.e) 
            geom.boolean_difference(externalcircle, internalcircle)
            mesh = geom.generate_mesh()
        return mesh

    def mesh(self):
        return self.triangular_mesh()

class PerfilI(Section):
    def __init__(self, b:float, h:float, t1:float, t2:float):
        super().__init__()
        m = b*t1/(h*t2)
        n = b/h
        pt1 = 12+72*m + 150*m**2 + 90*m**3
        pt2 = 11+66*m + 135*m**2 + 90*m**3
        pt3 = 10*n**2 * ((3+self.nu)*m + 3*m**2)
        self.k = 10*(1+self.nu)*(1+3*m)**2/(pt1 + self.nu * pt2 + pt3)

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
