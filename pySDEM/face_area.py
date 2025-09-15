import numpy as np

def face_area(f, v):
    """
    Compute the area of each face in a mesh.

    Parameters:
    f (ndarray): nf x 3 array of triangulations of a spherical surface mesh
    v (ndarray): nv x 3 array of vertex coordinates of a spherical surface mesh

    Returns:
    ndarray: nf x 1 array of face areas
    """
    v12 = v[f[:, 1], :] - v[f[:, 0], :]
    v23 = v[f[:, 2], :] - v[f[:, 1], :]
    v31 = v[f[:, 0], :] - v[f[:, 2], :]

    a = np.sqrt(np.sum(v12**2, axis=1))
    b = np.sqrt(np.sum(v23**2, axis=1))
    c = np.sqrt(np.sum(v31**2, axis=1))

    s = (a + b + c) / 2.0
    fa = np.sqrt(s * (s - a) * (s - b) * (s - c))

    return fa