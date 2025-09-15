import numpy as np
import pySTM
import pySDEM
import pyvista as pv
from scipy.special import spherical_jn, spherical_yn
import matplotlib.pyplot as plt


# Load in data ================================================================
loaded_data = pySTM.load_STM_results(r"C:\Users\105849\Desktop\scala_l4_with_errorL_80Lo.h5")


n_coeffs_I = loaded_data["metadata"]["computation_parameters"]["n_coeffs_I"]
STM = loaded_data["STM"]
G = loaded_data["results_data"]["G"]
lmax_O = int(loaded_data["metadata"]["user_settings"]["lmax_O"])
areas = loaded_data["mesh_data"]["EXTERNAL"]["mesh_metadata"]["areas"]   
frequencies = loaded_data["results_data"]["frequencies"]
S = loaded_data["mesh_data"]["EXTERNAL"]["SDEM_Coordinates"]
grid_gammaO = loaded_data["mesh_data"]["EXTERNAL"]["ExternalGrid"]
point_mapping = loaded_data["results_data"]["point_mappings"]["point_mapping"]
error = loaded_data["results_data"]["error_data"]["rel_error"]
x, y, z = S[:, 0], S[:, 1], S[:, 2]
r, lat, lon = pySDEM.cart_to_lat_lon(x, y, z)
# =============================================================================

# %%
excitation_response = np.zeros(n_coeffs_I,dtype=np.complex128)
excitation_response[0] = 1 #Y00
excitation_response[1] = 1 #Y1-1
excitation_response[2] = 2j #Y10
excitation_response[3] = 1-1j #Y11

synthesized_response_clm1d = np.einsum('ijk,i->jk', STM, excitation_response)
synthesized_response_clm = np.zeros((2, lmax_O + 1, lmax_O + 1, len(frequencies)), dtype=np.complex128)
synthesized_velocity = np.zeros([len(lat),len(frequencies)],dtype=np.complex128)

for f in range(len(frequencies)):
    for idl in range(lmax_O + 1):
        for idm in range(-idl, idl + 1):
            id1d = idl**2 + idl + idm
            if idm >= 0:
                synthesized_response_clm[0, idl, idm, f] = synthesized_response_clm1d[id1d, f]
            else:
                synthesized_response_clm[1, idl, abs(idm), f] = synthesized_response_clm1d[id1d, f]

rho = 1.3
C = 343
a = np.sqrt((np.sum(areas))/(4*np.pi))

def impedance_Z(l,ka,rho,C):
    # Not in fact true impedance! True impedance is 1j*rho*C*h1/dh1, but 
    # compensation is needed for the definition in  E.G. Williams
    # Fourier Acoustics eq. 6.93 p. 207
    d_jn = spherical_jn(l, ka, derivative=True)
    d_yn = spherical_yn(l, ka, derivative=True)
    dh1 = d_jn + 1j * d_yn
    return 1j*rho*C/dh1 


totalPower = np.zeros(len(frequencies))
totalPower_db = np.zeros(len(frequencies))
for id_f in range(len(frequencies)):
    k = 2*np.pi*frequencies[id_f]/C
    ka = k*a
    clm_v = synthesized_response_clm[:,:,:,id_f]
    clm_p = np.zeros_like(clm_v)
    for id_l in range(lmax_O):
        clm_p[:,id_l,:] = clm_v[:,id_l,:]*impedance_Z(id_l,ka,rho,C)
        clm_p_abs_sqr = np.sum(np.abs(clm_p)**2)
        contribution = 1/(2*rho*C*k**2)*clm_p_abs_sqr * 1/(np.sqrt((4*np.pi))) # 1/(np.sqrt((4*np.pi))) is Compensation for 4pi normalization used in pyshtools
        totalPower[id_f] += contribution
        
totalPower_db = 10 * np.log10(totalPower / 1E-12)

synthesized_velocity = G @ synthesized_response_clm1d
# %%
current_freq_idx = 0

# Create the power plot figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(frequencies, totalPower_db)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power (dB)')
ax.set_title('Total Power vs Frequency')

marker, = ax.plot(frequencies[current_freq_idx], totalPower_db[current_freq_idx], 'ro', markersize=8)
plotter = pv.Plotter()

def update_3d_plot(freq_idx):
    global current_freq_idx
    current_freq_idx = freq_idx

    plotter.clear()

    field = np.real(synthesized_velocity[:, freq_idx])
    mesh = grid_gammaO.copy()
    mesh.point_data['Velocity Magnitude'] = field

    plotter.add_mesh(mesh, scalars='Velocity Magnitude', cmap='turbo', show_edges=False)
    plotter.add_scalar_bar(title='Velocity Magnitude')
    plotter.add_text(f'Frequency: {frequencies[freq_idx]:.2f} Hz', position='upper_edge')
    plotter.reset_camera()
    plotter.render()

def on_click(event):
    if event.inaxes == ax:
        idx = np.abs(frequencies - event.xdata).argmin()
        marker.set_data([frequencies[idx]], [totalPower_db[idx]])
        fig.canvas.draw_idle()
        update_3d_plot(idx)

fig.canvas.mpl_connect('button_press_event', on_click)
update_3d_plot(current_freq_idx)
plt.show(block=False)
plotter.show()




# %%
# Plot the error for all points across frequencies in a scientific publication quality figure
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from itertools import cycle
# Set matplotlib style for publication quality
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 22
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8
plt.figure(figsize=(16, 6))
# Determine the maximum degree l_max based on the number of points
num_points = error.shape[0]
l_max = int((-1 + np.sqrt(1 + 4 * num_points)) / 2)  # Solve quadratic equation for l_max
# Define custom colors and create a cyclic iterator
colors = cycle(['#0FDBD7','#bf0000', '#005B96', '#92B268', '#14A693'])
# Create legend labels for each degree l (Y_{l,m})
legend_labels = [f'$Y_{{{l},m}}$' for l in range(l_max + 1)]
# Map each point index to its corresponding degree l
point_to_l = []
for l in range(l_max + 1):
    for m in range(-l, l + 1):
        point_to_l.append(l)
# Assign colors to each degree l
l_colors = [next(colors) for _ in range(l_max + 1)]
# Plot each point's error across frequencies, assigning same color to points with same l
for i in range(num_points):
    l = point_to_l[i]  # Get the degree l for this point
    label = legend_labels[l] if (i == sum((2 * k + 1) for k in range(l))) else None  # Label only the first m for each l
    plt.semilogy(frequencies, error[i, :], 
                 color=l_colors[l], alpha=0.8, label=label)

# Add horizontal dashed line at maximum error
max_error = np.max(error)
plt.axhline(y=max_error, color='red', linestyle='--', linewidth=2, 
            alpha=0.8, label=f'Max Error: {max_error:.3e}')

plt.grid(True, linestyle='-', alpha=0.7)
plt.xlabel('Frequency (Hz)', fontsize=20)
plt.ylabel('Error Magnitude', fontsize=20)
plt.title('Error vs Frequency, "Complex Geometry"', fontsize=22, pad=15)
plt.ylim([1E-2,0.2])
# Add horizontal legend below the figure
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=6, 
           fontsize=18, frameon=True, edgecolor='black')
# Improve layout
plt.tight_layout()
plt.savefig('error_analysis.pdf', dpi=300, bbox_inches='tight')
plt.show()
