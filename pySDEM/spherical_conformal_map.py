import numpy as np
from scipy.sparse import find, csc_array
from scipy.sparse.linalg import spsolve
from numpy.linalg import norm, solve
from numpy import cross
from pySDEM.cotangent_laplacian import cotangent_laplacian
from pySDEM.linear_beltrami_solver import linear_beltrami_solver
from pySDEM.beltrami_coefficient import beltrami_coefficient

def spherical_conformal_map(v, f):
    """
    Compute spherical conformal map for a genus-0 mesh

    Parameters:
    v: vertex coordinates (n x 3 numpy array)
    f: face indices (m x 3 numpy array)

    Returns:
    map: mapped vertex coordinates on sphere (n x 3 numpy array)
    """
    # Check whether the input mesh is genus-0
    if len(v) - 3*len(f)/2 + len(f) != 2:
        raise ValueError('The mesh is not a genus-0 closed surface.')

    # Find the most regular triangle as the "big triangle"
    temp = v[f.T.ravel('F')]
    e1 = np.sqrt(np.sum((temp[1::3] - temp[2::3])**2, axis=1))
    e2 = np.sqrt(np.sum((temp[0::3] - temp[2::3])**2, axis=1))
    e3 = np.sqrt(np.sum((temp[0::3] - temp[1::3])**2, axis=1))
    regularity = (np.abs(e1/(e1+e2+e3) - 1/3) +
                 np.abs(e2/(e1+e2+e3) - 1/3) +
                 np.abs(e3/(e1+e2+e3) - 1/3))
    bigtri = np.argmin(regularity)

    # North pole step: Compute spherical map by solving laplace equation
    nv = len(v)
    M = cotangent_laplacian(v, f)

    p1, p2, p3 = f[bigtri]
    fixed = np.array([p1, p2, p3])

    [mcol, mrow, mval] = find(M[fixed, :].T)
    M = M - csc_array((mval, (fixed[mrow], mcol)), shape=(nv, nv)) + csc_array((np.ones(len(fixed)), (fixed, fixed)), shape=(nv, nv))

    # Set boundary condition for big triangle
    x1, y1, x2, y2 = 0, 0, 1, 0
    a = v[p2] - v[p1]
    b = v[p3] - v[p1]
    sin1 = norm(cross(a, b))/norm(a)/norm(b)
    ori_h = norm(b)*sin1
    ratio = norm([x1-x2, y1-y2])/norm(a)
    y3 = ori_h*ratio
    x3 = np.sqrt(norm(b)**2 * ratio**2 - y3**2)

    # Solve Laplace equation
    c = np.zeros(nv)
    d = np.zeros(nv)
    c[[p1, p2, p3]] = [x1, x2, x3]
    d[[p1, p2, p3]] = [y1, y2, y3]

    # # Perform LU decomposition
    # lu = splu(M.astype(complex))
    # # Solve for the complex vector
    # z = lu.solve(c + 1j*d)
    B = c + 1j*d
    z = spsolve(M, B)
    # z = solve(M.todense(), B)
    z = z - np.mean(z)

    # Inverse stereographic projection
    S = np.column_stack([
        2*z.real/(1+abs(z)**2),
        2*z.imag/(1+abs(z)**2),
        (-1+abs(z)**2)/(1+abs(z)**2)
    ])

    # Find optimal big triangle size
    w = S[:, 0]/(1+S[:, 2]) + 1j*S[:, 1]/(1+S[:, 2])

    # Find southernmost triangle
    index = np.argsort(abs(z[f[:, 0]]) + abs(z[f[:, 1]]) + abs(z[f[:, 2]]))
    inner = index[1] if index[0] == bigtri else index[0]

    # Compute triangle sizes
    NorthTriSide = (abs(z[f[bigtri, 0]] - z[f[bigtri, 1]]) +
                    abs(z[f[bigtri, 1]] - z[f[bigtri, 2]]) +
                    abs(z[f[bigtri, 2]] - z[f[bigtri, 0]])) / 3

    SouthTriSide = (abs(w[f[inner, 0]] - w[f[inner, 1]]) +
                    abs(w[f[inner, 1]] - w[f[inner, 2]]) +
                    abs(w[f[inner, 2]] - w[f[inner, 0]])) / 3

    # Rescale
    z = z * np.sqrt(NorthTriSide * SouthTriSide) / NorthTriSide

    # Final inverse stereographic projection
    S = np.column_stack([
        2*z.real/(1+abs(z)**2),
        2*z.imag/(1+abs(z)**2),
        (-1+abs(z)**2)/(1+abs(z)**2)
    ])

    if np.any(np.isnan(S)):
        S = spherical_tutte_map(f, bigtri)

    # South pole step
    I = np.argsort(S[:, 2])
    fixnum = max(round(len(v)/10), 3)
    fixed = I[:min(len(v), fixnum)]

    # South pole stereographic projection
    P = np.column_stack([S[:, 0]/(1+S[:, 2]), S[:, 1]/(1+S[:, 2])])

    # Compute Beltrami coefficient and solve
    mu = beltrami_coefficient(P, f, v)
    map_coords = linear_beltrami_solver(P, f, mu, fixed, P[fixed])

    if np.any(np.isnan(map_coords)):
        fixnum = fixnum * 5
        fixed = I[:min(len(v), fixnum)]
        map_coords = linear_beltrami_solver(P, f, mu, fixed, P[fixed])

        if np.any(np.isnan(map_coords)):
            map_coords = P

    z = map_coords[:, 0] + 1j*map_coords[:, 1]

    # Final inverse south pole stereographic projection
    map_final = np.column_stack([
        2*z.real/(1+abs(z)**2),
        2*z.imag/(1+abs(z)**2),
        -(abs(z)**2-1)/(1+abs(z)**2)
    ])

    return map_final