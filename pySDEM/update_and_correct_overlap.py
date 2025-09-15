import numpy as np
from pySDEM.rotate_sphere import rotate_sphere
from pySDEM.stereographic_projection import stereographic_projection
from pySDEM.linear_beltrami_solver import linear_beltrami_solver
from pySDEM.beltrami_coefficient import beltrami_coefficient
from pySDEM.south_pole import south_pole

def update_and_correct_overlap(f, S, r, bigtri, dr, dt):
    delta = 0.1
    r_ori = r
    f_ori = f
    flag = True

    while flag:
        r = r_ori;
        # Update and normalize
        r = r_ori + dt * dr
        norms = np.linalg.norm(r, axis=1, keepdims=True)
        r = r / norms

        # North pole step
        # rotate_sphere
        S_rotN, _, _ = rotate_sphere(f, S, bigtri)
        r_rotN, _, RN_inv = rotate_sphere(f, r, bigtri)

        # stereographic_projection
        p_S_rotN = stereographic_projection(S_rotN)
        p_r_rotN = stereographic_projection(r_rotN)

        # Remove row bigtri
        f = np.delete(f, bigtri, axis=0)

        # Ignore outermost triangles
        I = np.argsort(-S_rotN[:, 2])
        ig_N = I[:max(round(len(S) / 10), 3)]
        mask_N = (np.isin(f[:, 0], ig_N) |
                  np.isin(f[:, 1], ig_N) |
                  np.isin(f[:, 2], ig_N))
        ignore_index_N = np.where(mask_N)[0]

        # beltrami_coefficient
        mu_N = beltrami_coefficient(p_S_rotN, f, p_r_rotN)
        overlap_N = np.setdiff1d(np.where(np.abs(mu_N) >= 1)[0], ignore_index_N)

        if len(overlap_N) == 0:
            r_newN = r_rotN
            north_success = True
        else:
            mu_N[overlap_N] = (1 - delta) * mu_N[overlap_N] / np.abs(mu_N[overlap_N])
            # linear_beltrami_solver
            p_lbsN = linear_beltrami_solver(p_S_rotN, f, mu_N, ig_N, p_r_rotN[ig_N])
            mu_N = beltrami_coefficient(p_S_rotN, f, p_lbsN)
            overlap_N = np.setdiff1d(np.where(np.abs(mu_N) >= 1)[0], ignore_index_N)
            if len(overlap_N) == 0:
                north_success = True
            else:
                dt /= 2
                north_success = False
            r_newN = stereographic_projection(p_lbsN)

        # Rotate sphere back
        r_newN = (RN_inv @ r_newN.T).T
        f = f_ori

        if north_success:

            south_f = south_pole(f, r, bigtri)

            S_rotS, _, _ = rotate_sphere(f, S, south_f)
            r_rotS, _, RS_inv = rotate_sphere(f, r_newN, south_f)

            p_S_rotS = stereographic_projection(S_rotS)
            p_r_rotS = stereographic_projection(r_rotS)

            f = np.delete(f, south_f, axis=0)

            I1 = np.argsort(-S_rotS[:, 2])
            ig_S = I1[:max(round(len(S) / 10), 3)]
            mask_S = (np.isin(f[:, 0], ig_S) |
                      np.isin(f[:, 1], ig_S) |
                      np.isin(f[:, 2], ig_S))
            ignore_index_S = np.where(mask_S)[0]

            mu_S = beltrami_coefficient(p_S_rotS, f, p_r_rotS)
            overlap_S = np.setdiff1d(np.where(np.abs(mu_S) >= 1)[0], ignore_index_S)

            if len(overlap_S) == 0:
                r_newS = r_rotS
                south_success = True
            else:
                mu_S[overlap_S] = (1 - delta) * mu_S[overlap_S] / np.abs(mu_S[overlap_S])
                p_lbsS = linear_beltrami_solver(p_S_rotS, f, mu_S, ig_S, p_r_rotS[ig_S])
                mu_S = beltrami_coefficient(p_S_rotS, f, p_lbsS)
                overlap_S = np.setdiff1d(np.where(np.abs(mu_S) >= 1)[0], ignore_index_S)
                if len(overlap_S) == 0:
                    south_success = True
                else:
                    dt /= 2
                    south_success = False
                r_newS = stereographic_projection(p_lbsS)

        if north_success and south_success:
            flag = False

        if dt < 1e-10:
            flag = False

    if dt < 1e-10:
        r_new = r_ori
    else:
        r_new = (RS_inv @ r_newS.T).T

    return r_new
