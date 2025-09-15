import numpy as np
def find_rotation_matrix(points1, points2):
    """
    Find the optimal rotation matrix that transforms points1 to align with points2.

    Parameters:
    points1: numpy array of shape (N, 3) containing the original points
    points2: numpy array of shape (N, 3) containing the transformed points

    Returns:
    R: The 3x3 rotation matrix
    rmsd: Root mean square deviation between the transformed points1 and points2
    """

    # Center the point clouds
    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)

    centered1 = points1 - centroid1
    centered2 = points2 - centroid2

    # Compute the covariance matrix
    H = centered1.T @ centered2

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)

    # Ensure right-handed coordinate system (handle reflection case)
    V = Vt.T
    det = np.linalg.det(V @ U.T)
    if det < 0:
        V[:, -1] *= -1

    # Calculate rotation matrix
    R = V @ U.T

    # Calculate RMSD to evaluate the quality of the fit
    transformed_points = (centered1 @ R) + centroid2
    rmsd = np.sqrt(np.mean(np.sum((transformed_points - points2) ** 2, axis=1)))

    return R, rmsd