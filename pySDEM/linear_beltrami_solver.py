import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np

def linear_beltrami_solver(v, f, mu, landmark, target):
    # Split up needed terms
    a = (1 - 2*np.real(mu) + np.abs(mu)**2) / (1 - np.abs(mu)**2)
    b = -2*np.imag(mu) / (1 - np.abs(mu)**2)
    g = (1 + 2*np.real(mu) + np.abs(mu)**2) / (1 - np.abs(mu)**2)
    # Triangle vertices
    f0, f1, f2 = f[:,0], f[:,1], f[:,2]
    # Edge vectors in 2D
    uxv0 = v[f1,1] - v[f2,1]
    uyv0 = v[f2,0] - v[f1,0]
    uxv1 = v[f2,1] - v[f0,1]
    uyv1 = v[f0,0] - v[f2,0]
    uxv2 = v[f0,1] - v[f1,1]
    uyv2 = v[f1,0] - v[f0,0]
    # Triangle areas
    L0 = np.sqrt(uxv0**2 + uyv0**2)
    L1 = np.sqrt(uxv1**2 + uyv1**2)
    L2 = np.sqrt(uxv2**2 + uyv2**2)
    s = 0.5*(L0 + L1 + L2)
    area = np.sqrt(s*(s-L0)*(s-L1)*(s-L2))
    # Building per-triangle matrix coefficients
    v00 = (a*uxv0*uxv0 + 2*b*uxv0*uyv0 + g*uyv0*uyv0) / area
    v11 = (a*uxv1*uxv1 + 2*b*uxv1*uyv1 + g*uyv1*uyv1) / area
    v22 = (a*uxv2*uxv2 + 2*b*uxv2*uyv2 + g*uyv2*uyv2) / area
    v01 = (a*uxv1*uxv0 + b*(uxv1*uyv0 + uxv0*uyv1) + g*uyv1*uyv0) / area
    v12 = (a*uxv2*uxv1 + b*(uxv2*uyv1 + uxv1*uyv2) + g*uyv2*uyv1) / area
    v20 = (a*uxv0*uxv2 + b*(uxv0*uyv2 + uxv2*uyv0) + g*uyv0*uyv2) / area

    # Assemble sparse matrix
    I = np.concatenate([f0, f1, f2, f0, f1, f1, f2, f2, f0])
    J = np.concatenate([f0, f1, f2, f1, f0, f2, f1, f0, f2])
    V = np.concatenate([v00, v11, v22, v01, v01, v12, v12, v20, v20]) / 2

    # Create initial matrix
    A = sp.csr_matrix(((-1)*V, (I, J)), shape=(v.shape[0], v.shape[0]))

    # Right-hand side
    tc = target[:,0] + 1j*target[:,1]
    rhs = -A[:,landmark].dot(tc)
    rhs[landmark] = tc

    # More efficient way to zero out landmark rows/columns
    mask = np.ones(A.shape[0], dtype=bool)
    mask[landmark] = False

    # Zero out rows and columns for landmarks
    A = A.multiply(mask.reshape(-1, 1))
    A = A.multiply(mask.reshape(1, -1))

    # Add ones on diagonal for landmarks
    A = A + sp.csr_matrix((np.ones(len(landmark)), (landmark, landmark)), shape=A.shape)

    # Solve system
    sol = spla.spsolve(A, rhs)
    return np.column_stack((sol.real, sol.imag))