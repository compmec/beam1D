import numpy as np

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Cannot normaliza a null vector")
    return v/norm

def projection(v:np.ndarray, w:np.ndarray) -> np.ndarray:
    return np.inner(v, w)*w/np.sum(w**2)

def random_between(a:float, b:float) -> float:
    if a == b:
        return a
    t = np.random.rand()
    return t*a + (1-t)*b

def random_vector(confs:list = None) -> np.ndarray:
    v = [random_between(-1, 1),
         random_between(-1, 1),
         random_between(-1, 1)]
    v = np.array(v)
    if confs is None:
        return v
    if len(confs) != 3:
        raise ValueError("parameter must be a list of len = 3 with bools")
    for i, c in enumerate(confs):
        if not c:
            v[i] = 0
    return v

def random_unit_vector(confs:list=None) -> np.ndarray:
    v = random_vector(confs)
    return normalize(v)
