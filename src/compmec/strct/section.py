import numpy as np
import pygmsh
from typing import Optional
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
        super().__init__(nu)
        self.b = b
        self.h = h

    def shear_coefficient(self):
        return 10*(1+self.nu)/(12+11*self.nu)

    def compute_areas(self):
        k = self.shear_coefficient()
        self.Ax = self.b*self.h
        self.Ay = k * self.Ax
        self.Az = k * self.Ax

    def compute_inertias(self):
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
        super().__init__(nu)
        self.be = be
        self.he = he
        self.bi = bi
        self.hi = hi

class ThinRetangular(Section):
    def __init__(self, b: float, h:float, nu:float):
        """
        b is the lower base
        h is the height
        """
        super().__init__(nu)
        self.b = b
        self.h = h

class Square(Section):
    def __doc__(self):
        """
        Docs
        """
    def __init__(self, b:float, nu:float):
        super().__init__(nu)
        self.b = b

    def shear_coefficient(self):
        return 20*(1+self.nu)/(4+3*self.nu)

    def compute_areas(self):
        k = self.shear_coefficient()
        self.Ax = self.b**2
        self.Ay = k * self.Ax
        self.Az = k * self.Ax
        print("Warning: Areas for a square is not yet well defined")
        # raise NotImplementedError("Areas for a square are not defined")

    def compute_inertias(self):
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
        super().__init__(nu)
        self.be = be
        self.bi = bi


class ThinSquare(Section):
    def __doc__(self):
        """
        Docs
        """
    def __init__(self, b:float, nu:float):
        super().__init__(nu)
        self.b = b

class Circle(Section):
    def __init__(self, R:float, nu:float):
        super().__init__(nu)
        self.R = R

    @property
    def R(self):
        return self.__R

    @R.setter
    def R(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError("The radius cannot be zero or negative")
        self.__R = value

    def shear_coefficient(self):
        return 6*(1+self.nu)/(7+6*self.nu)

    def compute_areas(self):
        k = self.shear_coefficient()
        self.Ax = np.pi* self.R**2
        self.Ay = k * self.Ax
        self.Az = k * self.Ax
    
    def compute_inertias(self):
        R4 = self.R**4
        self.Ix = np.pi * R4 / 2
        self.Iy = np.pi * R4 / 4
        self.Iz = np.pi * R4 / 4
    
    def triangular_mesh(self, meshsize:float):
        with pygmsh.geo.Geometry() as geom:
            geom.add_circle((0, 0), self.R, mesh_size=meshsize)
            mesh = geom.generate_mesh()
        return mesh


class HollowCircle(Circle):
    def __init__(self, Ri: float, Re: float, nu: float):
        R = (Ri+Re)/2
        super().__init__(R=R, nu=nu)
        self.Ri = Ri
        self.Re = Re
        self.e = self.Re-self.Ri
        
    @property
    def Ri(self):
        return self.__Ri

    @property
    def Re(self):
        return self.__Re

    @property
    def e(self):
        return self.__e
    
    @Ri.setter
    def Ri(self, value: float):
        value = float(value)
        if value < 0:
            raise ValueError("The internal radius must be non-negative")
        if value == 0:
            raise ValueError("If the internal radius is 0, use Circle instead")
        self.__Ri = value
    
    @Re.setter
    def Re(self, value: float):
        value = float(value)
        if value <= 0:
            raise ValueError("The external radius must be positive")
        self.__Re = value

    @e.setter
    def e(self, value: float):
        value = float(value)
        if value <= 0:
            raise ValueError("The thickness must be positive")
        self.__e = value

    def shear_coefficient(self):
        m2 = (self.Ri/self.Re)**2
        nu = self.nu
        return 6*(1+nu)/( (7+6*nu) + 4*m2*(5+3*nu)/(1+m2)**2)

    def compute_areas(self):
        k = self.shear_coefficient()
        self.Ax = np.pi* (self.Re**2 - self.Ri**2)
        self.Ay = k * self.Ax
        self.Az = k * self.Ax

    def compute_inertias(self):
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

class ThinCircle(HollowCircle):
    __ratiomax = 0.2
    __ratiodefault = 0.01
    def __init__(self, R:float, nu:float, e: Optional[float]=None):
        """
        Creates a thin circle section.
        * R is the mean radius.
        * nu is the poisson coefficient.
        * e is optional thickness.
            If not given, it's 0.01*R
        """
        print("Received :")
        print("R = ", R)
        print("nu = ", nu)
        print("e = ", e)
        if e is None:
            e = ThinCircle.__ratiodefault*R
        if e/R > ThinCircle.__ratiomax:
            raise ValueError(f"The ratio e/R is too big. Received e/R = %.3f > %.3f" % (e/self.R, ThinCircle.__ratiomax))
        Ri = R - 0.5*e
        Re = R + 0.5*e
        super().__init__(Ri=Ri, Re=Re, nu=nu)

    def shear_coefficient(self):
        return 2*(1+self.nu)/(4+3*self.nu)
        
    def compute_areas(self):
        k = self.shear_coefficient()
        self.Ax = 2 * np.pi * self.R * self.e
        self.Ay = k * self.Ax
        self.Az = k * self.Ax
        
    def compute_inertias(self):
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
    def __init__(self, b: float, h: float, t1: float, t2: float, nu: float):
        super().__init__(nu)
        self.b = b
        self.h = h
        self.t1 = t1
        self.t2 = t2
        
    def shear_coefficient(self):
        n = self.b/self.h
        m = n*self.t1/self.t2
        pt1 = 12+72*m + 150*m**2 + 90*m**3
        pt2 = 11+66*m + 135*m**2 + 90*m**3
        pt3 = 10*n**2 * ((3+self.nu)*m + 3*m**2)
        numerador = 10*(1+self.nu)*(1+3*m)**2
        denominador = (pt1 + self.nu * pt2 + pt3)
        return numerador/denominador

class General(Section):
    def __init__(self, curves: list, nu: float):
        """
        curves is a list of closed curves that defines the geometry
        Each curve is a Nurbs, with the points.
        It's possible to have a circle, only with one curve, a circle
        Until now, it's not implemented
        """
        super().__init__(nu)
        raise Exception("Not implemented")
    


def main():
    pass

if __name__ == "__main__":
    main()
