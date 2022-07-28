from compmec.strct.element import EulerBernoulli
import numpy as np
import pytest

@pytest.mark.order(1)
@pytest.mark.dependency()
def test_begin():
	pass

@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_EulerBernoulli():
    ntests = 10
    for i in range(ntests):
        p0 = np.random.rand(3)
        p1 = np.random.rand(3)
        t = np.linspace(0, 1, 10)
        beam = EulerBernoulli([p0, p1])
        for ti in t:
            pgood = (1-ti)*p0 + ti*p1
            ptest = beam.path(ti)
            np.testing.assert_allclose(ptest, pgood)

@pytest.mark.order(1)
@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin", "test_EulerBernoulli"])
def test_end():
	pass
