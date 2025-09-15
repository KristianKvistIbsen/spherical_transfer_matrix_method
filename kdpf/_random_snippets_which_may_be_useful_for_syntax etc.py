from ansys.dpf import core as dpf
import numpy as np

# =============================================================================
# # Metadata Utilities
# =============================================================================



# =============================================================================
# # DPF Field Extractors
# =============================================================================
def get_displacement_from_mesh(model,mesh,tfreq):
    op_displacements = dpf.operators.result.displacement()
    op_displacements.inputs.time_scoping.connect(tfreq)
    op_displacements.inputs.mesh_scoping.connect(mesh.nodes.scoping)
    op_displacements.inputs.data_sources.connect(model)
    op_displacements.inputs.mesh.connect(mesh)
    return op_displacements.outputs.fields_container()



def get_velocity_from_mesh(model,mesh,tfreq):

    mesh_scoping = get_scoping_from_mesh(mesh,'Nodal')

    # Get xyz velocities at all nodal locations on skin mesh
    op_velocity = dpf.operators.result.velocity()
    op_velocity.inputs.time_scoping.connect(tfreq)
    op_velocity.inputs.mesh_scoping.connect(mesh_scoping)
    op_velocity.inputs.data_sources.connect(model)
    op_velocity.inputs.mesh.connect(mesh)

    return op_velocity.outputs.fields_container()
def get_stress_from_mesh(model,mesh,tfreq):

    mesh_scoping = get_scoping_from_mesh(mesh,'Nodal')

    op_stress = dpf.operators.result.stress() # operator instantiation
    op_stress.inputs.time_scoping.connect(tfreq)# optional
    op_stress.inputs.mesh_scoping.connect(mesh_scoping)# optional
    op_stress.inputs.data_sources.connect(model)
    op_stress.inputs.mesh.connect(mesh)# optional

    return op_stress.outputs.fields_container()
def get_normal_velocities(model,skin_mesh,tfreq,skin_mesh_scoping,nodal_normals):
    # Get xyz velocities at all nodal locations on skin mesh
    op_velocities = dpf.operators.result.velocity()
    op_velocities.inputs.time_scoping.connect(tfreq)
    op_velocities.inputs.mesh_scoping.connect(skin_mesh_scoping)
    op_velocities.inputs.data_sources.connect(model)
    op_velocities.inputs.mesh.connect(skin_mesh)

    # Get nodal normal velocities on skin mesh by dot product + misc field info
    opDot = dpf.operators.math.generalized_inner_product_fc()
    opDot.inputs.field_or_fields_container_A.connect(op_velocities)
    opDot.inputs.field_or_fields_container_B.connect(nodal_normals)
    return opDot.outputs.fields_container()

def get_normal_velocitiy_fc_from_skin_mesh(model,skin_mesh,tfreq,skin_mesh_scoping,nodal_normals):
    # Get xyz velocities at all nodal locations on skin mesh
    op_velocities = dpf.operators.result.velocity()
    op_velocities.inputs.time_scoping.connect(tfreq)
    op_velocities.inputs.mesh_scoping.connect(skin_mesh_scoping)
    op_velocities.inputs.data_sources.connect(model)
    op_velocities.inputs.mesh.connect(skin_mesh)

    # Get nodal normal velocities on skin mesh by dot product + misc field info
    opDot = dpf.operators.math.generalized_inner_product_fc()
    opDot.inputs.field_or_fields_container_A.connect(op_velocities)
    opDot.inputs.field_or_fields_container_B.connect(nodal_normals)
    return opDot

# def get_modal_normal_displacement_fc_from_skin_mesh(model,skin_mesh,tfreq,skin_mesh_scoping,nodal_normals):
#     op_disp = dpf.operators.result.displacement() # operator instantiation
#     op_disp.inputs.time_scoping.connect(tfreq)# optional
#     op_disp.inputs.mesh_scoping.connect(skin_mesh_scoping)# optional
#     op_disp.inputs.data_sources.connect(model)
#     op_disp.inputs.mesh.connect(skin_mesh)# optional
#     # my_fields_container = op_disp.outputs.fields_container()

#     # Get nodal normal velocities on skin mesh by dot product + misc field info
#     opDot = dpf.operators.math.generalized_inner_product_fc()
#     opDot.inputs.field_or_fields_container_A.connect(op_disp)
#     opDot.inputs.field_or_fields_container_B.connect(nodal_normals)
#     return opDot

def get_modal_normal_displacement_fc_from_skin_mesh(model, skin_mesh, tfreq, skin_mesh_scoping, nodal_normals):
    # Get displacement field for positive frequencies
    op_disp = dpf.operators.result.displacement()
    op_disp.inputs.time_scoping.connect(tfreq)
    op_disp.inputs.mesh_scoping.connect(skin_mesh_scoping)
    op_disp.inputs.data_sources.connect(model)
    op_disp.inputs.mesh.connect(skin_mesh)

    # Project displacements onto normal directions
    op_dot = dpf.operators.math.generalized_inner_product_fc()
    op_dot.inputs.field_or_fields_container_A.connect(op_disp)
    op_dot.inputs.field_or_fields_container_B.connect(nodal_normals)

    return op_dot

def get_normal_displacements(model,skin_mesh,tfreq,skin_mesh_scoping,nodal_normals):
    op_displacements = dpf.operators.result.displacement()
    op_displacements.inputs.time_scoping.connect(tfreq)
    op_displacements.inputs.mesh_scoping.connect(skin_mesh_scoping)
    op_displacements.inputs.data_sources.connect(model)
    op_displacements.inputs.mesh.connect(skin_mesh)

    opDot = dpf.operators.math.generalized_inner_product_fc()
    opDot.inputs.field_or_fields_container_A.connect(op_displacements)
    opDot.inputs.field_or_fields_container_B.connect(nodal_normals)
    return opDot.outputs.fields_container()
# def get_elastic_strain_energy_density(model,mesh,tfreq):


def bandpass_tfreq(tfreq,tf_min,tf_max):
    op_bandpass = dpf.operators.filter.timefreq_band_pass() # operator instantiation
    op_bandpass.inputs.time_freq_support.connect(tfreq)
    op_bandpass.inputs.min_threshold.connect(float(tf_min))
    op_bandpass.inputs.max_threshold.connect(float(tf_max))
    my_time_freq_support = op_bandpass.outputs.time_freq_support()

    return my_time_freq_support



# =============================================================================
# Compute workflows
# =============================================================================


# =============================================================================
# # Export Utilities
# =============================================================================





