import numpy as np
from scipy.sparse import csr_matrix
from pySDEM.face_area import face_area

def f2v_area(v, f):
    nv = len(v)
    nf = len(f)

    # find area
    area = face_area(f, v)

    # create matrix
    row = np.hstack((f[:, 2], f[:, 0], f[:, 1]))
    col = np.hstack((np.arange(nf), np.arange(nf), np.arange(nf)))
    val = np.hstack((area, area, area))
    M = csr_matrix((val, (row, col)), shape=(nv, nf))

    # normalize
    vertex_area_sum = np.array(M.sum(axis=1)).flatten()
    M = M.multiply(1.0 / vertex_area_sum[:, np.newaxis])

    return M