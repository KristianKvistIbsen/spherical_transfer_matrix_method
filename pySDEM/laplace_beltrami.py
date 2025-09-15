import numpy as np
from scipy.sparse import coo_matrix

def laplace_beltrami(v, f):
    nv = v.shape[0]
    f1, f2, f3 = f[:,0], f[:,1], f[:,2]

    # Calculate edge lengths similar to MATLAB version
    l = np.column_stack([
        np.sqrt(np.sum((v[f2] - v[f3])**2, axis=1)),
        np.sqrt(np.sum((v[f3] - v[f1])**2, axis=1)),
        np.sqrt(np.sum((v[f1] - v[f2])**2, axis=1))
    ])
    l1, l2, l3 = l[:,0], l[:,1], l[:,2]

    s = 0.5 * (l1 + l2 + l3)
    area = np.sqrt(s * (s - l1) * (s - l2) * (s - l3))

    cot12 = (l1**2 + l2**2 - l3**2) / (4.0 * area)
    cot23 = (l2**2 + l3**2 - l1**2) / (4.0 * area)
    cot31 = (l1**2 + l3**2 - l2**2) / (4.0 * area)

    II = np.concatenate((f1, f2, f2, f3, f3, f1, f1, f2, f3))
    JJ = np.concatenate((f2, f1, f3, f2, f1, f3, f1, f2, f3))
    V = np.concatenate((
        -cot12, -cot12, -cot23, -cot23, -cot31, -cot31,
        (cot12 + cot31), (cot12 + cot23), (cot31 + cot23)
    )) / 2.0

    L = coo_matrix((V, (II, JJ)), shape=(nv, nv))
    return L