import numpy as np
from scipy.sparse import coo_matrix

def regular_triangle(f, v):
    """
    Find the triangle with the most regular 1-ring neighborhood.

    Parameters:
    f (ndarray): nf x 3 array of triangulations of a spherical surface mesh
    v (ndarray): nv x 3 array of vertex coordinates of a spherical surface mesh

    Returns:
    int: Index of the triangle with the most regular 1-ring neighborhood
    """
    nv = len(v)
    nf = len(f)

    # face regularity
    temp = v[f.flatten(), :3]
    e1 = np.sqrt(np.sum((temp[1::3, :3] - temp[2::3, :3])**2, axis=1))
    e2 = np.sqrt(np.sum((temp[0::3, :3] - temp[2::3, :3])**2, axis=1))
    e3 = np.sqrt(np.sum((temp[0::3, :3] - temp[1::3, :3])**2, axis=1))
    R_f = np.abs(e1 / (e1 + e2 + e3) - 1/3) + \
          np.abs(e2 / (e1 + e2 + e3) - 1/3) + \
          np.abs(e3 / (e1 + e2 + e3) - 1/3)

    # create face-to-vertex matrix
    row = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
    col = np.concatenate([np.arange(nf), np.arange(nf), np.arange(nf)])
    val = (1/3) * np.ones(3 * nf)
    H = coo_matrix((val, (row, col)), shape=(nv, nf))

    # vertex regularity
    R_v = H.dot(R_f)

    # average vertex regularity for each face
    R_average = (1/3) * (R_v[f[:, 0]] + R_v[f[:, 1]] + R_v[f[:, 2]])
    regular_triangle = np.argmin(R_average)

    return regular_triangle