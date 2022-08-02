import numpy as np
from compmec.nurbs import SplineBaseFunction, SplineCurve
from compmec.nurbs.knotoperations import insert_knot_basefunction, insert_knot_controlpoints
# from compmec.strct.element import EulerBernoulli

class EulerBernoulliPos(object):


    @staticmethod
    def displacement(element, result: np.ndarray) -> SplineCurve:
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


            
            