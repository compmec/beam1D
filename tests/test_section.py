from compmec.strct.profile import Circle, HollowCircle, ThinCircle
import pytest
import numpy as np

TOLERANCE = 1e-9
PI = np.pi

@pytest.mark.order(1)
@pytest.mark.skip(reason="Due change from section to profile, some change on test must be made")
@pytest.mark.dependency()
def test_begin():
	pass


@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_circle():
    ntests = 10
    for i in range(ntests):
        R = 1+np.random.rand()
        nu = 0.01 + 0.48*np.random.rand()
        circle = Circle(R=R)
        coef = 6*(1+nu)/(7+6*nu)
        Ax = PI * R**2
        Iy = PI * R**4/4
        assert abs(circle.R - R) < TOLERANCE
        assert abs(circle.A[0] - Ax) < TOLERANCE
        assert abs(circle.A[1] - coef * Ax) < TOLERANCE
        assert abs(circle.A[2] - coef * Ax) < TOLERANCE
        assert abs(circle.I[0] - 2*Iy) < TOLERANCE
        assert abs(circle.I[1] - Iy) < TOLERANCE
        assert abs(circle.I[2] - Iy) < TOLERANCE

@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_hollowcircle():
    ntests = 10
    for i in range(ntests):
        R = 1+np.random.rand()
        e = 0.2*np.random.rand()
        Ri = R-0.5*e
        Re = R+0.5*e
        nu = 0.01 + 0.48*np.random.rand()
        m2 = (Ri/Re)**2
        coef = 6*(1+nu)*(1+m2)**2/((7+6*nu)*(1+m2)**2 + 4*m2*(5+3*nu))
        hollowcircle = HollowCircle(Ri=Ri, Re=Re)
        Ax = PI * (Re**2-Ri**2)
        Iy = PI * (Re**4-Ri**4)/4
        assert abs(hollowcircle.R - R) < TOLERANCE
        assert abs(hollowcircle.e - e) < TOLERANCE
        assert abs(hollowcircle.Ri - Ri) < TOLERANCE
        assert abs(hollowcircle.Re - Re) < TOLERANCE
        assert abs(hollowcircle.A[0] - Ax) < TOLERANCE 
        assert abs(hollowcircle.A[1] - coef * Ax) < TOLERANCE
        assert abs(hollowcircle.A[2] - coef * Ax) < TOLERANCE
        assert abs(hollowcircle.I[0] - 2*Iy) < TOLERANCE
        assert abs(hollowcircle.I[1] - Iy) < TOLERANCE
        assert abs(hollowcircle.I[2] - Iy) < TOLERANCE

@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_thincircle():
    ntests = 1
    for i in range(ntests):
        R = 1+np.random.rand()
        e = 0.01*R
        Ri, Re = R-0.5*e, R+0.5*e
        nu = 0.01 + 0.48*np.random.rand()
        coef = 2*(1+nu)/(4+3*nu)
        thincircle = ThinCircle(R=R, e=e)
        Ax = 2*PI*e*R
        Iy = PI * e * R**3
        assert abs(thincircle.R - R) < TOLERANCE
        assert abs(thincircle.e - e) < TOLERANCE
        assert abs(thincircle.Ri - Ri) < TOLERANCE
        assert abs(thincircle.Re - Re) < TOLERANCE
        assert abs(thincircle.A[0] - Ax) < TOLERANCE 
        assert abs(thincircle.A[1] - coef * Ax) < TOLERANCE
        assert abs(thincircle.A[2] - coef * Ax) < TOLERANCE
        assert abs(thincircle.I[0] - 2*Iy) < TOLERANCE
        assert abs(thincircle.I[1] - Iy) < TOLERANCE
        assert abs(thincircle.I[2] - Iy) < TOLERANCE


@pytest.mark.order(1)
@pytest.mark.dependency(depends=["test_begin",
                                 "test_circle", "test_hollowcircle", "test_thincircle"])
def test_end():
	pass




def main():
    test_begin()
    test_circle()
    test_hollowcircle()
    test_thincircle()
    test_end()

if __name__ == "__main__":
    main()