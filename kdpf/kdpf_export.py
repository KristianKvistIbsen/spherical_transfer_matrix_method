from ansys.dpf import core as dpf
import numpy as np

def generate_nodal_csv(mesh, field_data, export_path=None, user_header=None):
    nodal_coordinates = mesh.nodes.coordinates_field.data

    result = {
        'coordinates': nodal_coordinates,
        'values': field_data
    }

    # Export to CSV if path is provided
    if export_path:
        data = np.column_stack((nodal_coordinates, field_data))
        
        # Save to CSV using numpy
        np.savetxt(
            export_path,
            data,
            delimiter=',',
            header=','.join(user_header),
            comments='',  # This prevents the '#' character before the header
            fmt='%.16e'    # Format to 6 decimal places
        )

    return result


def field_to_csv(field, export_path=None, user_header=None):

    if field.location != "Nodal":
        field = field.to_nodal()

    nodal_coordinates = field.meshed_region.nodes.coordinates_field.data
    mapping = field.meshed_region.nodes.mapping_id_to_index

    # Initialize arrays
    n_nodes = field.meshed_region.nodes.n_nodes
    mapped_values = np.zeros(n_nodes)
    # Map thermal conductivity values to correct indices using the mapping
    for node_id, node_index in mapping.items():
        mapped_values[node_index] = field.get_entity_data_by_id(node_id)


    result = {
        'coordinates': nodal_coordinates,
        'values': mapped_values
    }

    # Export to CSV if path is provided
    if export_path:
        # Create header and data structure
        if user_header:
            header = user_header
        data = np.column_stack((nodal_coordinates, mapped_values))

        # Save to CSV using numpy
        np.savetxt(
            export_path,
            data,
            delimiter=',',
            header=','.join(header),
            comments='',  # This prevents the '#' character before the header
            fmt='%.16e'    # Format to 6 decimal places
        )

    return result

def field_container_to_csv(container, export_path=None):
    
    first_field = container[0]
    mesh = first_field.meshed_region
    
    # Extract nodal coordinates
    nodal_coordinates = mesh.nodes.coordinates_field.data
    mapping = mesh.nodes.mapping_id_to_index
    n_nodes = mesh.nodes.n_nodes
    
    # Process each field in the container
    n_fields = len(container)
    mapped_values_list = []
    field_names = []
    
    for i in range(n_fields):
        field = container[i]
        # Generate field name from label space
        label_dict = container.get_label_space(i)
        label_str = '_'.join([f"{k}_{v}" for k, v in sorted(label_dict.items())])
        field_name = f"field_{label_str}"
        field_names.append(field_name)
        
        # Ensure field is nodal
        if field.location != "Nodal":
            field = field.to_nodal()
        
        # Map field values to nodal indices
        mapped_values = np.zeros(n_nodes)
        for node_id, node_index in mapping.items():
            mapped_values[node_index] = field.get_entity_data_by_id(node_id)
        mapped_values_list.append(mapped_values)
    
    # Combine coordinates and field values
    data = np.column_stack([nodal_coordinates[:, 0], 
                           nodal_coordinates[:, 1], 
                           nodal_coordinates[:, 2]] + mapped_values_list)
    
    # Create header
    header = ['x', 'y', 'z'] + field_names
    
    # Export to CSV if path provided
    if export_path:
        np.savetxt(
            export_path,
            data,
            delimiter=',',
            header=','.join(header),
            comments='',
            fmt='%.16e'
        )
    
    # Return results
    return {
        'coordinates': nodal_coordinates,
        'values': mapped_values_list,
        'field_names': field_names
    }


def export_field_to_vtk(mesh,field,path):
    op_export = dpf.operators.serialization.vtk_export() # operator instantiation
    op_export.inputs.file_path.connect(path)
    op_export.inputs.mesh.connect(mesh)
    op_export.inputs.fields1.connect(field)
    op_export.run()

def export_mesh_to_vtk(mesh,path):
    op_export = dpf.operators.serialization.vtk_export() # operator instantiation
    op_export.inputs.file_path.connect(path)
    op_export.inputs.mesh.connect(mesh)
    op_export.run()