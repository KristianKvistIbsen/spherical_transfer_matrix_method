from scipy.sparse import coo_matrix
import numpy as np
def beltrami_coefficient(vertices, faces, mapped):
    nf = faces.shape[0]
    Mi = np.repeat(np.arange(nf), 3)
    Mj = faces.flatten()

    # Edge vectors in 2D
    e1 = vertices[faces[:,2], :2] - vertices[faces[:,1], :2]
    e2 = vertices[faces[:,0], :2] - vertices[faces[:,2], :2]
    e3 = vertices[faces[:,1], :2] - vertices[faces[:,0], :2]

    # Signed area
    area = 0.5 * (e1[:,0]*e2[:,1] - e1[:,1]*e2[:,0])
    # area_tile = np.tile(area, 3)

    # Build sparse derivative matrices
    Mx = np.ravel((np.column_stack([e1[:,1], e2[:,1], e3[:,1]]) / (2*area[:,None]))).flatten()
    My = -np.ravel((np.column_stack([e1[:,0], e2[:,0], e3[:,0]]) / (2*area[:,None]))).flatten()
    Dx = coo_matrix((Mx, (Mi, Mj)), shape=(nf, vertices.shape[0]))
    Dy = coo_matrix((My, (Mi, Mj)), shape=(nf, vertices.shape[0]))

    if mapped.shape[1] == 3:
        dXdu = Dx.dot(mapped[:,0]); dXdv = Dy.dot(mapped[:,0])
        dYdu = Dx.dot(mapped[:,1]); dYdv = Dy.dot(mapped[:,1])
        dZdu = Dx.dot(mapped[:,2]); dZdv = Dy.dot(mapped[:,2])
        E = dXdu**2 + dYdu**2 + dZdu**2
        G = dXdv**2 + dYdv**2 + dZdv**2
        F = dXdu*dXdv + dYdu*dYdv + dZdu*dZdv
        denom = E + G + 2*np.sqrt(np.maximum(E*G - F**2, 0))
        denom[denom == 0] = 1e-16
        mu = (E - G + 2j*F) / denom
    else:
        z = mapped[:,0] + 1j*mapped[:,1]
        Dz = (Dx - 1j*Dy).dot(z) / 2
        Dc = (Dx + 1j*Dy).dot(z) / 2
        denom = Dz
        denom[denom == 0] = 1e-16
        mu = Dc / denom
        mu[~np.isfinite(mu)] = 1

    return mu