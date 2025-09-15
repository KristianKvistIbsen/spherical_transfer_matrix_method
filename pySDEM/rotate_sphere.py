import numpy as np
def rotate_sphere(f, v, north_f):

    # Find the averaged center of the specified triangle and normalize it
    tri_center = (v[f[north_f, 0]] + v[f[north_f, 1]] + v[f[north_f, 2]]) / 3.0
    tri_center /= np.linalg.norm(tri_center)

    # Rotation about z-axis
    denom_z = np.sqrt(tri_center[0]**2 + tri_center[1]**2)
    sin_z = -tri_center[1] / denom_z
    cos_z =  tri_center[0] / denom_z
    Rz = np.array([[cos_z, -sin_z, 0],
                    [sin_z,  cos_z, 0],
                    [0,      0,     1]], dtype=float)

    # Apply rotation around z to get new coords
    B = Rz @ tri_center
    denom_y = np.sqrt(B[0]**2 + B[2]**2)
    sin_y = -B[0] / denom_y
    cos_y =  B[2] / denom_y
    Ry = np.array([[cos_y,  0, sin_y],
                    [0,      1,     0],
                    [-sin_y, 0, cos_y]], dtype=float)

    # Composite rotation matrix and its inverse
    M = Ry @ Rz
    Minv = Rz.T @ Ry.T

    # Rotate the entire set of vertices
    v_rot = v @ M.T

    return v_rot, M, Minv