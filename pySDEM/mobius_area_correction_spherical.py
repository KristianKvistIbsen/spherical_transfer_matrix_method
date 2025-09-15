import numpy as np
from scipy.optimize import minimize

def mobius_area_correction_spherical(v, f, map_coords):

    # Compute the area with normalization
    area_v = face_area(f, v)
    area_v = area_v / np.sum(area_v)

    # Project the sphere onto the plane
    p = stereographic(map_coords)
    z = p[:, 0] + 1j * p[:, 1]

    def area_map(x):
        # Calculate complex Mobius transformation
        numerator = (x[0] + x[1]*1j)*z + (x[2] + x[3]*1j)
        denominator = (x[4] + x[5]*1j)*z + (x[6] + x[7]*1j)
        fz = numerator / denominator

        # Convert back to coordinates
        coords = np.column_stack((np.real(fz), np.imag(fz)))
        coords_3d = stereographic(coords)

        areas = face_area(f, coords_3d)
        return areas / np.sum(areas)

    def d_area(x):
        # Objective function: mean(abs(log(area_map/area_v)))
        areas = area_map(x)
        ratios = areas / area_v
        return finite_mean(np.abs(np.log(ratios)))

    # Optimization setup
    x0 = np.array([1, 0, 0, 0, 0, 0, 1, 0])  # initial guess
    bounds = [(-100, 100)] * 8  # bounds for parameters

    # Optimization
    result = minimize(d_area, x0, method='L-BFGS-B', bounds=bounds)
    x = result.x

    # Obtain the conformal parameterization with area distortion corrected
    fz = ((x[0] + x[1]*1j)*z + (x[2] + x[3]*1j)) / ((x[4] + x[5]*1j)*z + (x[6] + x[7]*1j))
    map_mobius = stereographic(np.column_stack((np.real(fz), np.imag(fz))))

    return map_mobius, x

def face_area(f, v):
    """Compute the area of every face of a triangle mesh."""
    v12 = v[f[:, 1]] - v[f[:, 0]]
    v23 = v[f[:, 2]] - v[f[:, 1]]
    v31 = v[f[:, 0]] - v[f[:, 2]]

    a = np.sqrt(np.sum(v12 * v12, axis=1))
    b = np.sqrt(np.sum(v23 * v23, axis=1))
    c = np.sqrt(np.sum(v31 * v31, axis=1))

    s = (a + b + c) / 2
    return np.sqrt(s * (s-a) * (s-b) * (s-c))

def finite_mean(A):
    """Calculate mean avoiding Inf values."""
    return np.mean(A[np.isfinite(A)])

def stereographic(u):
    """
    Stereographic projection.
    For N-by-2 matrix, projects points in plane to sphere.
    For N-by-3 matrix, projects points on sphere to plane.
    """
    if u.shape[1] == 1:
        u = np.column_stack((np.real(u), np.imag(u)))

    x = u[:, 0]
    y = u[:, 1]

    if u.shape[1] < 3:
        z = 1 + x**2 + y**2
        return np.column_stack((2*x/z, 2*y/z, (-1 + x**2 + y**2)/z))
    else:
        z = u[:, 2]
        return np.column_stack((x/(1-z), y/(1-z)))