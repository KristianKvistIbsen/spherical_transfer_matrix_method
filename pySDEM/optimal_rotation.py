import numpy as np
from scipy.spatial.distance import cdist

def optimal_rotation(v, S, n_sample_points=50, random_seed=None):
    """
    Compute the optimal rotation matrix that best aligns transformed points S to original points v.
    Uses the Kabsch algorithm to find the rotation matrix that minimizes RMSD.
    
    Parameters:
    -----------
    v : numpy.ndarray
        Original node locations (N x 3)
    S : numpy.ndarray  
        Transformed node locations (N x 3)
    n_sample_points : int, optional
        Number of random points to use for computation. If None, uses all points.
    random_seed : int, optional
        Random seed for reproducible point selection
        
    Returns:
    --------
    R : numpy.ndarray
        3x3 rotation matrix that best aligns S to v
    rmsd : float
        Root mean square deviation after optimal rotation
    """
    
    # Convert to numpy arrays if needed
    v = np.array(v)
    S = np.array(S)
    
    # Ensure we have 3D points
    if v.shape[1] != 3 or S.shape[1] != 3:
        raise ValueError("Points must be 3D (N x 3 arrays)")
    
    if v.shape[0] != S.shape[0]:
        raise ValueError("v and S must have the same number of points")
    
    # Select random sample of points if specified
    if n_sample_points is not None and n_sample_points < len(v):
        if random_seed is not None:
            np.random.seed(random_seed)
        
        indices = np.random.choice(len(v), size=n_sample_points, replace=False)
        v_sample = v[indices]
        S_sample = S[indices]
    else:
        v_sample = v
        S_sample = S
    
    # Center both point sets (remove translation)
    v_centroid = np.mean(v_sample, axis=0)
    S_centroid = np.mean(S_sample, axis=0)
    
    v_centered = v_sample - v_centroid
    S_centered = S_sample - S_centroid
    
    # Compute covariance matrix H = S^T * v
    H = S_centered.T @ v_centered
    
    # Singular Value Decomposition
    U, _, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1, not -1 for reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute RMSD after rotation
    S_rotated = (S_sample - S_centroid) @ R.T + v_centroid
    rmsd = np.sqrt(np.mean(np.sum((v_sample - S_rotated)**2, axis=1)))
    
    return R, rmsd