import numpy as np
from scipy.sparse import coo_matrix

def cotangent_laplacian(v, f):
    """
    Compute the cotangent Laplacian of a mesh.
    
    Parameters:
        v: vertex coordinates (nv x 3 numpy array)
        f: faces (nf x 3 numpy array, integer indices)
    
    Returns:
        L: sparse cotangent Laplacian matrix (nv x nv)
    
    Reference:
    [1] P. T. Choi, K. C. Lam, and L. M. Lui, 
        "FLASH: Fast Landmark Aligned Spherical Harmonic Parameterization for Genus-0 Closed Brain Surfaces."
        SIAM Journal on Imaging Sciences, vol. 8, no. 1, pp. 67-94, 2015.
    """
    nv = len(v)
    
    # Extract vertex indices for each face
    f1, f2, f3 = f[:, 0], f[:, 1], f[:, 2]
    
    # Compute edge lengths
    l1 = np.sqrt(np.sum((v[f2] - v[f3])**2, axis=1))
    l2 = np.sqrt(np.sum((v[f3] - v[f1])**2, axis=1))
    l3 = np.sqrt(np.sum((v[f1] - v[f2])**2, axis=1))
    
    # Compute area using semi-perimeter formula
    s = (l1 + l2 + l3) * 0.5
    area = np.sqrt(s * (s-l1) * (s-l2) * (s-l3))
    
    # Compute cotangents
    cot12 = (l1**2 + l2**2 - l3**2) / (2 * area)
    cot23 = (l2**2 + l3**2 - l1**2) / (2 * area)
    cot31 = (l1**2 + l3**2 - l2**2) / (2 * area)
    
    # Compute diagonal elements
    diag1 = -cot12 - cot31
    diag2 = -cot12 - cot23
    diag3 = -cot31 - cot23
    
    # Construct sparse matrix
    II = np.concatenate([f1, f2, f2, f3, f3, f1, f1, f2, f3])
    JJ = np.concatenate([f2, f1, f3, f2, f1, f3, f1, f2, f3])
    V = np.concatenate([cot12, cot12, cot23, cot23, cot31, cot31, diag1, diag2, diag3])
    
    L = coo_matrix((V, (II, JJ)), shape=(nv, nv)).tocsr()
    
    return L