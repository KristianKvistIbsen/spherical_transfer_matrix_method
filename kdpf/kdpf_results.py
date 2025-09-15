from ansys.dpf import core as dpf
import numpy as np


def get_tfreq(model):
    return model.metadata.time_freq_support.time_frequencies


def get_normal_velocities(model,skin_mesh,tfreq,nodal_normals):
    # Get xyz velocities at all nodal locations on skin mesh
    op_velocities = dpf.operators.result.velocity()
    op_velocities.inputs.time_scoping.connect(tfreq)
    op_velocities.inputs.mesh_scoping.connect(skin_mesh.nodes.scoping)
    op_velocities.inputs.data_sources.connect(model)
    op_velocities.inputs.mesh.connect(skin_mesh)

    # Get nodal normal velocities on skin mesh by dot product + misc field info
    opDot = dpf.operators.math.generalized_inner_product_fc()
    opDot.inputs.field_or_fields_container_A.connect(op_velocities)
    opDot.inputs.field_or_fields_container_B.connect(nodal_normals)
    return opDot.outputs.fields_container()