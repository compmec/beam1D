import numpy as np
import pytest
from usefulfunc import *

@pytest.mark.dependency()
def test_begin():
    pass

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_normalize():
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            for c in (-1, 0, 1):
                v = np.array([a, b, c])
                if a == 0 and b == 0 and c == 0:
                    continue
                vtest = normalize(v)
                if a*b*c:
                    vgood = v/np.sqrt(3)
                elif (a*b) or (b*c) or (a*c):
                    vgood = v/np.sqrt(2)
                else:
                    vgood = v
                np.testing.assert_almost_equal(vtest, vgood)

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_begin"])
def test_random_between():
    ntests = 1000
    for i in range(ntests):
        a = 2
        b = 5
        m = random_between(a, b)
        assert a <= m
        assert m <= b

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_random_between"])
def test_random_vector():
    v = random_vector()
    v = random_vector([True, True, True])
    v = random_vector([True, True, False])
    assert v[2] == 0
    v = random_vector([True, False, True])
    assert v[1] == 0
    v = random_vector([False, True, True])
    assert v[0] == 0
    v = random_vector([True, False, False])
    assert v[1] == 0
    assert v[2] == 0
    v = random_vector([False, True, False])
    assert v[0] == 0
    assert v[2] == 0
    v = random_vector([False, False, True])
    assert v[0] == 0
    assert v[1] == 0
    

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_random_vector"])
def test_random_unit_vector():
    ntests = 100
    tolerance = 1e-15
    for i in range(ntests):
        v = random_unit_vector()
        assert np.abs(np.sum(v**2) - 1) < tolerance
        v = random_unit_vector([True, True, True])
        assert np.abs(np.sum(v**2) - 1) < tolerance
        v = random_unit_vector([True, True, False])
        assert v[2] == 0
        assert np.abs(np.sum(v**2) - 1) < tolerance
        v = random_unit_vector([True, False, True])
        assert v[1] == 0
        assert np.abs(np.sum(v**2) - 1) < tolerance
        v = random_unit_vector([False, True, True])
        assert v[0] == 0
        assert np.abs(np.sum(v**2) - 1) < tolerance
        v = random_unit_vector([True, False, False])
        assert v[1] == 0
        assert v[2] == 0
        assert np.abs(np.sum(v**2) - 1) < tolerance
        v = random_unit_vector([False, True, False])
        assert v[0] == 0
        assert v[2] == 0
        assert np.abs(np.sum(v**2) - 1) < tolerance
        v = random_unit_vector([False, False, True])
        assert v[0] == 0
        assert v[1] == 0
        assert np.abs(np.sum(v**2) - 1) < tolerance

@pytest.mark.timeout(2)
@pytest.mark.dependency(depends=["test_random_vector"])
def test_projection():
    ntests = 100
    tolerance = 1e-15
    for i in range(ntests):
        v = random_vector()
        w = random_vector()
        v -= projection(v, w)
        assert np.abs(np.inner(v, w)) < tolerance




if __name__ == "__main__":
    pytest.main()