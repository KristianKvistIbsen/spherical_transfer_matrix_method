from ansys.dpf import core as dpf
import trimesh

def get_normals(skin_mesh, location="Nodal", asOperator=False):

    op_get_normals = dpf.operators.geo.normals_provider_nl()
    op_get_normals.inputs.mesh.connect(skin_mesh)
    op_get_normals.inputs.requested_location.connect(location)

    if asOperator:
        return op_get_normals
    else:    
        return op_get_normals.outputs.field()
    
def get_areas(skin_mesh, location="Nodal", asOperator=False):
    op = dpf.operators.geo.elements_volume() 
    op.inputs.mesh.connect(skin_mesh)
    areas_elem_x2 = op.outputs.field()
    areas_elem_x2.meshed_region = skin_mesh
    
    op_scale = dpf.operators.math.scale()
    op_scale.inputs.field.connect(areas_elem_x2)
    op_scale.inputs.ponderation.connect(0.5)
    
    areas_elem = op_scale.outputs.field()
    
    if location == "Elemental":
        if asOperator:
            return op
        else:    
            return areas_elem
    else:
        op = dpf.operators.averaging.to_nodal()
        op.inputs.field.connect(areas_elem)
        areas_node = op.outputs.field()
        if asOperator:
            return op
        else:    
            return areas_node


def stl_to_dpf_mesh(stl_path):
    mesh = trimesh.load(stl_path, file_type='stl')
    connectivity = mesh.faces
    coordinates = mesh.vertices
    meshed_region = dpf.MeshedRegion(
        num_nodes=coordinates.shape[0],
        num_elements=connectivity.shape[0]
    )
    print("Warning: scaling factor of 1000 used to convert from mm to m")
    for i, coord in enumerate(coordinates, start=0):
        meshed_region.nodes.add_node(i, coord / 1000)

    for i, face in enumerate(connectivity, start=0):
        meshed_region.elements.add_element(i, "shell", face.tolist())
    return meshed_region, mesh.area_faces