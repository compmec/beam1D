import numpy as np


class Material(object):
	def __init__(self):
		super().__init__(self)

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

    def triangular_mesh(self, elementsize:float):
        raise NotImplementedError("This function must be redefined by child class")

    def mesh(self, elementsize:float = None):
        if elementsize is None:
            elementsize = 0.1*np.sqrt(self.Ax)
        return self.triangular_mesh(elementsize)


class Structural1D(object):
    def __init__(self, path):
        if isinstance(path, (tuple, list)):
            p0 = np.array(path[0]) 
            p1 = np.array(path[1]) 
            self._path = lambda t: (1-t)*p0 + t*p1 
        elif callable(path):
             self._path = path
        else:
            raise TypeError("Not expected received argument")
        p0 = np.array(self.path(0)) 
        p1 = np.array(self.path(1)) 
        self._p = p1 - p0 
        self._L = np.sqrt(np.sum(self._p**2))

    def path(self, t:float) -> np.ndarray:
        if t < 0 or t > 1:
            raise ValueError("t in path must be in [0, 1]")
        return self._path(t)

    @property
    def p(self) -> np.ndarray:
        return self._p

    @property
    def L(self) -> float:
        return self._L

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

    def rotation_matrix33(self):
        px, py, pz = self.p
        cos = px/self.L
        pyz = py**2 + pz**2
        if cos == 1:
            return np.eye(3)
        elif cos == -1:
            return -np.eye(3)
        R33 = np.array([[0, 0, 0],
                        [0, pz**2, -py*pz],
                        [0, -py*pz, py**2]], dtype="float64")
        R33 *= (1-cos)/pyz
        R33 += np.array([[px, py, pz],
                         [-py, px, 0],
                         [-pz, 0, px]])/self.L
        return R33

    