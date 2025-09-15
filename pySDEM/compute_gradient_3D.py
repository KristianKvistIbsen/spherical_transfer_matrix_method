import numpy as np

def compute_gradient_3D(v, f, g):
    # compute edges
    e1 = v[f[:, 2], :] - v[f[:, 1], :]
    e2 = v[f[:, 0], :] - v[f[:, 2], :]
    e3 = v[f[:, 1], :] - v[f[:, 0], :]

    # compute area
    cross12 = np.cross(e1, e2)
    area = np.abs(0.5 * np.sqrt(np.sum(cross12**2, axis=1)))
    N = (1.0 / (2 * area))[:, np.newaxis] * cross12

    # compute gradient
    temp = (g[f[:, 0], np.newaxis] * e1 +
            g[f[:, 1], np.newaxis] * e2 +
            g[f[:, 2], np.newaxis] * e3)
    grad = np.cross(N, temp)
    grad = (1.0 / (2 * area))[:, np.newaxis] * grad

    return grad