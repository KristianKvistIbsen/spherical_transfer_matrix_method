import numpy as np
from scipy.sparse.linalg import spsolve
from pySDEM.regular_triangle import regular_triangle
from pySDEM.face_area import face_area
from pySDEM.f2v_area import f2v_area
from pySDEM.update_and_correct_overlap import update_and_correct_overlap
from pySDEM.compute_gradient_3D import compute_gradient_3D
from pySDEM.laplace_beltrami import laplace_beltrami
from pySDEM.lumped_mass_matrix import lumped_mass_matrix
from pySDEM.spherical_conformal_map import spherical_conformal_map
from pySDEM.mobius_area_correction_spherical import mobius_area_correction_spherical
from pySDEM.find_rotation_matrix import find_rotation_matrix

def SphericalDensityEqualizingMap(v, f, population, dt=0.1, epsilon=1e-3, max_iter=200):

    S1 = spherical_conformal_map(v,f)
    S, _  = mobius_area_correction_spherical(v,f,S1)

    r = np.array([
        S[:, 0] / np.sqrt(np.sum(S**2, axis=1)),
        S[:, 1] / np.sqrt(np.sum(S**2, axis=1)),
        S[:, 2] / np.sqrt(np.sum(S**2, axis=1))
    ]).T

    bigtri = regular_triangle(f, r)

    rho_f = population / face_area(f, r)
    rho_v = f2v_area(r, f) * rho_f

    step = 0
    rho_v_error = np.std(rho_v) / np.mean(rho_v)
    print('\n\nSDEM itt. \t \t std(rho)/mean(rho)')
    print(f'{step} \t \t \t \t {rho_v_error}')

    error_history = []

    while rho_v_error >= epsilon and step < max_iter:
        L = laplace_beltrami(r, f)
        A = lumped_mass_matrix(r, f)
        rho_v_temp = spsolve(A + dt * L, A @ rho_v)

        grad_rho_temp_f = compute_gradient_3D(r, f, rho_v_temp)
        grad_rho_temp_v = f2v_area(r, f) * grad_rho_temp_f

        dr = -np.column_stack((
            grad_rho_temp_v[:, 0] / rho_v_temp,
            grad_rho_temp_v[:, 1] / rho_v_temp,
            grad_rho_temp_v[:, 2] / rho_v_temp
        ))
        dr_proj = dr - np.sum(dr * r, axis=1)[:, np.newaxis] * r

        r = update_and_correct_overlap(f, S, r, bigtri, dr_proj, dt)

        step += 1
        rho_v_error = np.std(rho_v_temp) / np.mean(rho_v_temp)
        print(f'{step} \t \t \t \t {rho_v_error}')

        rho_f = population / face_area(f, r)
        rho_v = f2v_area(r, f) * rho_f

        error_history.append(rho_v_error)
        if len(error_history) > 3:
            error_history.pop(0)
        if len(error_history) == 3 and (max(error_history) - min(error_history)) < epsilon:
            print('Breaking out of SDEM loop: Solution Stagnation\n\n')
            break

    R, _ = find_rotation_matrix(v,r)
    r = r @ R

    return r

