from ansys.dpf import core as dpf

def get_skin_mesh_from_ns(NS, model, location="Nodal", asOperator=False):
    # Get Mesh Scoping based on named selection
    opNamedSel = dpf.operators.scoping.on_named_selection()
    opNamedSel.inputs.named_selection_name.connect(NS)
    opNamedSel.inputs.int_inclusive.connect(0)
    opNamedSel.inputs.data_sources.connect(model)
    opNamedSel.inputs.requested_location.connect(location)
    op_skin_mesh = dpf.operators.mesh.skin()
    op_skin_mesh.inputs.mesh.connect(model.metadata.meshed_region)
    op_skin_mesh.inputs.mesh_scoping.connect(opNamedSel)
    
    if asOperator:
        return op_skin_mesh
    else:    
        return op_skin_mesh.outputs.mesh()
    
def get_interface_named_selections(model):

    NS_contact = [NS for NS in model.metadata.available_named_selections if NS.endswith('CONTACT')]
    NS_target = [NS for NS in model.metadata.available_named_selections if NS.endswith('TARGET')]

    return NS_contact, NS_target
