import numpy as np
from compmec.nurbs import SplineBaseFunction, SplineCurve
from compmec.nurbs.knotoperations import insert_knot_basefunction, insert_knot_controlpoints
from compmec.strct.__classes__ import Structural1D

class EulerBernoulliPos(object):

    __INSTANCE = None

    def __init__(self):
        EulerBernoulliPos.__INSTANCE = self
        self.__NAME2FUNCTIONS = {"L2norm(u)": self.L2norm_displacement,
                                 "u": self.displacement,
                                 "ux": self.displacement_x,
                                 "uy": self.displacement_y,
                                 "uz": self.displacement_z}
    
    @staticmethod
    def getInstance():
        if EulerBernoulliPos.__INSTANCE is None:
            EulerBernoulliPos()
        return EulerBernoulliPos.__INSTANCE
        
    @staticmethod
    def field(fieldname: str, element: Structural1D, result: np.ndarray) -> SplineCurve:
        if not isinstance(element, Structural1D):
            raise TypeError(f"The given element must be a Structural1D instance. Received {type(element)}")
        instance = EulerBernoulliPos.getInstance()
        if fieldname not in instance.__NAME2FUNCTIONS.keys():
            raise ValueError(f"Received argument is not valid. They are {list(instance.__NAME2FUNCTIONS.keys())}")
        function = instance.__NAME2FUNCTIONS[fieldname]
        curve = function(element, result)
        return curve

    

    def displacement(self, element, result: np.ndarray) -> SplineCurve:
        # if not isinstance(element, EulerBernoulli):
        #     raise TypeError("Element must be a EulerBernoulli instance")
        result = np.array(result)
        if result.ndim != 2:
            raise ValueError("result must be a 2D array")
        npts, dofs = result.shape
        if npts != len(element.ts):
            raise ValueError("The number result.shape[0] must be equal to element.ts")
        
        p = 3
        U = [0, 0]
        for i, ti in enumerate(element.ts):
            U += 2*[ti]
        U += [1, 1]
        N = SplineBaseFunction(U)
        dN = N.derivate()
        
        Ctrlpts = np.zeros((N.n, 3))
        M = np.zeros((N.n, N.n))
        B = np.zeros((N.n, 3))
        for i, ti in enumerate(element.ts):
            M[2*i, :] = N(ti)
            M[2*i+1, :] = dN(ti)
            n = element.normal(ti)
            B[2*i, :] = result[i, :3]
            B[2*i+1, :] = np.cross(result[i, 3:], n)
            
        Ctrlpts = np.linalg.solve(M, B)
        for i, ti in enumerate(element.ts):
            if ti == 0:
                continue
            if ti == 1:
                continue
            Ctrlpts = insert_knot_controlpoints(N, Ctrlpts, ti)
            N = insert_knot_basefunction(N, ti)
        curve = SplineCurve(N, Ctrlpts)
        return curve

    def L2norm_displacement(self, element, result: np.ndarray) -> SplineCurve:
        disp = self.displacement(element, result)
        knots = []
        for i, ui in enumerate(disp.U):
            if ui not in knots:
                knots.append(ui)
        usample = []
        for i in range(len(knots)-1):
            usample += list(np.linspace(knots[i], knots[i+1], disp.F.p+2, endpoint=False))
        usample.append(knots[-1])
        B = np.zeros(len(usample))
        L = np.zeros((disp.npts, len(usample)))
        for i, ui in enumerate(usample):
            L[:, i] = disp.F[:](ui)
            B[i] = np.sqrt(np.sum(disp(ui)**2))
        newControlPoints = np.linalg.solve(L @ L.T, L @ B)
        newControlPoints = newControlPoints.reshape((len(newControlPoints), 1))
        return disp.__class__(disp.F, newControlPoints)

    def displacement_x(self, element, result: np.ndarray) -> SplineCurve:
        displace = self.displacement(element, result)
        P = np.zeros((displace.P.shape[0], 1))
        P[:, 0] = displace.P[:, 0]
        return displace.__class__(displace.F, P)

    def displacement_y(self, element, result: np.ndarray) -> SplineCurve:
        displace = self.displacement(element, result)
        P = np.zeros((displace.P.shape[0], 1))
        P[:, 0] = displace.P[:, 1]
        return displace.__class__(displace.F, P)

    def displacement_z(self, element, result: np.ndarray) -> SplineCurve:
        displace = self.displacement(element, result)
        P = np.zeros((displace.P.shape[0], 1))
        P[:, 0] = displace.P[:, 2]
        return displace.__class__(displace.F, P)
        