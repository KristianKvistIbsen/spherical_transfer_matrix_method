import numpy as np
def stereographic_projection(v):

    # If input has three columns, do stereographic projection (3D -> 2D).
    if v.shape[1] == 3:
        denom = 1 - v[:, 2]
        p = np.column_stack((v[:, 0] / denom, v[:, 1] / denom))
        p[~np.isfinite(p)] = np.inf
    else:
        # If input is Nx1 and complex, convert to real/imag.
        if v.shape[1] == 1 and np.iscomplexobj(v):
            real_part = np.real(v[:, 0])
            imag_part = np.imag(v[:, 0])
            v = np.column_stack((real_part, imag_part))
        # Inverse stereographic projection (2D -> 3D).
        z = 1 + v[:, 0]**2 + v[:, 1]**2
        p = np.column_stack((
            2 * v[:, 0] / z,
            2 * v[:, 1] / z,
            (v[:, 0]**2 + v[:, 1]**2 - 1) / z
        ))
        invalid = ~np.isfinite(z) | np.isnan(z)
        p[invalid] = [0, 0, 1]
    return p