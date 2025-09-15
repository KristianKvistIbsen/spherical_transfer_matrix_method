import numpy as np

def south_pole(f, v, bigtri):
    f_center = (v[f[:, 0]] + v[f[:, 1]] + v[f[:, 2]]) / 3
    radius = np.linalg.norm(f_center, axis=1)
    f_center = f_center / radius[:, None]
    dist_sq = np.sum((f_center - f_center[bigtri,0])**2, axis=1)
    return np.argmax(dist_sq)