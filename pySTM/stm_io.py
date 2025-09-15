import h5py
import numpy as np
from datetime import datetime
import warnings
import os

def package_STM_results(STM, mesh_data, metadata, results_data, output_file="STM_computation_results.h5"):
    """
    Package STM computation results into a compressed HDF5 file.
    
    Parameters:
    -----------
    STM : numpy.ndarray
        Complex STM matrix with shape (n_coeffs_I, n_coeffs_O, n_frequencies)
    mesh_data : dict
        Dictionary containing mesh information for INTERNAL, EXTERNAL, and FULL_MESH
    metadata : dict
        Dictionary containing user settings, computation parameters, and frequency data
    results_data : dict
        Dictionary containing STM, frequencies, export files info, spherical harmonics, and point mappings
    output_file : str
        Output filename for the HDF5 file
    
    Returns:
    --------
    str : Path to the created HDF5 file
    """
    
    def save_grid_data(group, grid_obj, grid_name):
        """Helper function to save grid/mesh data"""
        try:
            # Save points
            if hasattr(grid_obj, 'points'):
                group.create_dataset(f"{grid_name}/points", 
                                   data=grid_obj.points, 
                                   compression='gzip', compression_opts=9)
            
            # Save cells/connectivity
            if hasattr(grid_obj, 'cells_dict') and grid_obj.cells_dict:
                cells_group = group.create_group(f"{grid_name}/cells")
                for cell_type, cell_data in grid_obj.cells_dict.items():
                    cells_group.create_dataset(str(cell_type), 
                                             data=cell_data, 
                                             compression='gzip', compression_opts=9)
            elif hasattr(grid_obj, 'cells'):
                group.create_dataset(f"{grid_name}/cells", 
                                   data=grid_obj.cells, 
                                   compression='gzip', compression_opts=9)
            
            # Save additional grid properties
            if hasattr(grid_obj, 'n_points'):
                group.attrs[f"{grid_name}_n_points"] = grid_obj.n_points
            if hasattr(grid_obj, 'n_cells'):
                group.attrs[f"{grid_name}_n_cells"] = grid_obj.n_cells
                
            # Save field data if available
            if hasattr(grid_obj, 'field_data') and grid_obj.field_data:
                field_group = group.create_group(f"{grid_name}/field_data")
                for field_name, field_data in grid_obj.field_data.items():
                    if isinstance(field_data, np.ndarray):
                        field_group.create_dataset(field_name, 
                                                 data=field_data, 
                                                 compression='gzip', compression_opts=9)
            
            # Save array data (like Area)
            if hasattr(grid_obj, '__getitem__'):
                try:
                    # Try to get common field names
                    common_fields = ['Area', 'Volume', 'Quality']
                    for field in common_fields:
                        try:
                            field_data = grid_obj[field]
                            if isinstance(field_data, np.ndarray):
                                group.create_dataset(f"{grid_name}/{field}", 
                                                   data=field_data, 
                                                   compression='gzip', compression_opts=9)
                        except (KeyError, TypeError):
                            continue
                except:
                    pass
                    
        except Exception as e:
            warnings.warn(f"Could not save grid data for {grid_name}: {str(e)}")
    
    def save_dict_recursively(group, data_dict, path=""):
        """Recursively save dictionary data to HDF5 group"""
        for key, value in data_dict.items():
            current_path = f"{path}/{key}" if path else key
            
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                save_dict_recursively(subgroup, value, current_path)
            
            elif isinstance(value, np.ndarray):
                if np.iscomplexobj(value):
                    # Save complex arrays as compound dataset
                    complex_dtype = np.dtype([('real', value.real.dtype), ('imag', value.imag.dtype)])
                    complex_data = np.empty(value.shape, dtype=complex_dtype)
                    complex_data['real'] = value.real
                    complex_data['imag'] = value.imag
                    group.create_dataset(key, data=complex_data, 
                                       compression='gzip', compression_opts=9)
                else:
                    group.create_dataset(key, data=value, 
                                       compression='gzip', compression_opts=9)
            
            elif hasattr(value, 'data') and isinstance(value.data, np.ndarray):
                # Handle DPFArray or similar objects
                group.create_dataset(key, data=value.data, 
                                   compression='gzip', compression_opts=9)
                group.attrs[f"{key}_type"] = str(type(value).__name__)
            
            elif isinstance(value, (list, tuple)):
                try:
                    # Try to convert to numpy array
                    arr_value = np.array(value)
                    group.create_dataset(key, data=arr_value, 
                                       compression='gzip', compression_opts=9)
                except:
                    # If conversion fails, save as string
                    group.attrs[key] = str(value)
            
            elif isinstance(value, (str, int, float, bool, type(None))):
                if value is None:
                    group.attrs[key] = "None"
                else:
                    group.attrs[key] = value
            
            else:
                # For other types, try to serialize or convert to string
                try:
                    group.attrs[key] = str(value)
                except:
                    warnings.warn(f"Could not save {current_path} of type {type(value)}")
    
    # Create HDF5 file with compression
    with h5py.File(output_file, 'w') as f:
        # Add file metadata
        f.attrs['creation_date'] = datetime.now().isoformat()
        f.attrs['file_format'] = 'STM_Results_HDF5'
        f.attrs['version'] = '1.0'
        
        # Save STM matrix (main result)
        STM_group = f.create_group('STM_Matrix')
        if np.iscomplexobj(STM):
            complex_dtype = np.dtype([('real', STM.real.dtype), ('imag', STM.imag.dtype)])
            complex_data = np.empty(STM.shape, dtype=complex_dtype)
            complex_data['real'] = STM.real
            complex_data['imag'] = STM.imag
            STM_group.create_dataset('STM', data=complex_data, 
                                   compression='gzip', compression_opts=9)
        else:
            STM_group.create_dataset('STM', data=STM, 
                                   compression='gzip', compression_opts=9)
        
        STM_group.attrs['shape'] = STM.shape
        STM_group.attrs['dtype'] = str(STM.dtype)
        
        # Save mesh data
        mesh_group = f.create_group('Mesh_Data')
        
        for mesh_type, mesh_info in mesh_data.items():
            mesh_subgroup = mesh_group.create_group(mesh_type)
            
            for data_name, data_value in mesh_info.items():
                if data_name.endswith('Grid') or data_name.endswith('Mesh'):
                    # Handle grid/mesh objects
                    save_grid_data(mesh_subgroup, data_value, data_name)
                elif isinstance(data_value, np.ndarray):
                    # Handle numpy arrays (like SDEM coordinates)
                    mesh_subgroup.create_dataset(data_name, data=data_value, 
                                               compression='gzip', compression_opts=9)
                elif isinstance(data_value, dict):
                    # Handle nested dictionaries (like mesh_metadata)
                    save_dict_recursively(mesh_subgroup.create_group(data_name), data_value)
                else:
                    try:
                        mesh_subgroup.attrs[data_name] = str(data_value)
                    except:
                        warnings.warn(f"Could not save mesh data: {data_name}")
        
        # Save metadata
        metadata_group = f.create_group('Metadata')
        save_dict_recursively(metadata_group, metadata)
        
        # Save results data
        results_group = f.create_group('Results_Data')
        save_dict_recursively(results_group, results_data)
        
        # Save compression information
        f.attrs['compression'] = 'gzip'
        f.attrs['compression_level'] = 9
        
        # Calculate and save file statistics
        f.attrs['n_groups'] = len(f.keys())
        
        def count_datasets(group):
            count = 0
            for item in group.values():
                if isinstance(item, h5py.Dataset):
                    count += 1
                elif isinstance(item, h5py.Group):
                    count += count_datasets(item)
            return count
        
        f.attrs['n_datasets'] = count_datasets(f)
    
    print(f"STM results successfully packaged to: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    return output_file


def load_STM_results(input_file):
    """
    Load STM computation results from HDF5 file.
    
    Parameters:
    -----------
    input_file : str
        Path to the HDF5 file containing STM results
    
    Returns:
    --------
    dict : Dictionary containing all loaded data with reconstructed grid objects
    """
    import pyvista as pv
    
    def load_complex_dataset(dataset):
        """Helper to load complex datasets"""
        if dataset.dtype.names and 'real' in dataset.dtype.names and 'imag' in dataset.dtype.names:
            return dataset['real'][:] + 1j * dataset['imag'][:]
        else:
            return dataset[:]
    
    def reconstruct_grid(grid_group):
        """Reconstruct PyVista grid object from HDF5 data"""
        try:
            # Get points
            points = None
            cells = None
            cell_types = None
            
            for key, item in grid_group.items():
                if key == 'points' and isinstance(item, h5py.Dataset):
                    points = item[:]
                elif key == 'cells' and isinstance(item, h5py.Group):
                    # Handle cells stored as dictionary
                    cells_dict = {}
                    for cell_type_name, cell_data in item.items():
                        if isinstance(cell_data, h5py.Dataset):
                            cells_dict[int(cell_type_name)] = cell_data[:]
                    
                    # Convert cells_dict to PyVista format
                    if cells_dict:
                        # Assume the most common cell type or first available
                        first_cell_type = list(cells_dict.keys())[0]
                        cells = cells_dict[first_cell_type]
                        
                        # Determine PyVista cell type
                        if first_cell_type == 5:  # VTK_TRIANGLE
                            cell_types = [5] * len(cells)
                        elif first_cell_type == 9:  # VTK_QUAD
                            cell_types = [9] * len(cells)
                        elif first_cell_type == 10:  # VTK_TETRA
                            cell_types = [10] * len(cells)
                        else:
                            cell_types = [first_cell_type] * len(cells)
                
                elif key == 'cells' and isinstance(item, h5py.Dataset):
                    # Handle cells stored as array
                    cells = item[:]
            
            if points is not None:
                if cells is not None:
                    # Create unstructured grid
                    if cell_types is None:
                        # Assume triangular cells if not specified
                        n_cells = len(cells)
                        cell_types = [5] * n_cells  # VTK_TRIANGLE
                    
                    # Format cells for PyVista (add cell size as first element)
                    formatted_cells = []
                    for i, cell in enumerate(cells):
                        formatted_cells.extend([len(cell)] + list(cell))
                    
                    grid = pv.UnstructuredGrid(formatted_cells, cell_types, points)
                else:
                    # Create point cloud if no cells
                    grid = pv.PolyData(points)
                
                # Add field data if available
                for key, item in grid_group.items():
                    if key in ['Area', 'Volume', 'Quality'] and isinstance(item, h5py.Dataset):
                        data = item[:]
                        if len(data) == grid.n_cells:
                            grid.cell_data[key] = data
                        elif len(data) == grid.n_points:
                            grid.point_data[key] = data
                
                return grid
            else:
                return None
                
        except Exception as e:
            warnings.warn(f"Could not reconstruct grid: {str(e)}")
            return None
    
    def load_group_recursively(group):
        """Recursively load HDF5 group to dictionary"""
        result = {}
        
        # Load attributes
        for attr_name, attr_value in group.attrs.items():
            if isinstance(attr_value, bytes):
                attr_value = attr_value.decode('utf-8')
            if attr_value == "None":
                attr_value = None
            result[attr_name] = attr_value
        
        # Load datasets and subgroups
        for key, item in group.items():
            if isinstance(item, h5py.Dataset):
                result[key] = load_complex_dataset(item)
            elif isinstance(item, h5py.Group):
                # Check if this group contains grid data
                if any(subkey in ['points', 'cells'] for subkey in item.keys()):
                    reconstructed_grid = reconstruct_grid(item)
                    if reconstructed_grid is not None:
                        result[key] = reconstructed_grid
                    else:
                        result[key] = load_group_recursively(item)
                else:
                    result[key] = load_group_recursively(item)
        
        return result
    
    with h5py.File(input_file, 'r') as f:
        # Load main STM matrix
        STM_data = load_complex_dataset(f['STM_Matrix/STM'])
        
        # Load mesh data with grid reconstruction
        mesh_data = {}
        if 'Mesh_Data' in f:
            for mesh_type, mesh_group in f['Mesh_Data'].items():
                mesh_data[mesh_type] = {}
                for data_name, data_item in mesh_group.items():
                    if isinstance(data_item, h5py.Group):
                        # Check if this is a grid group
                        if any(subkey in ['points', 'cells'] for subkey in data_item.keys()):
                            reconstructed_grid = reconstruct_grid(data_item)
                            if reconstructed_grid is not None:
                                mesh_data[mesh_type][data_name] = reconstructed_grid
                            else:
                                mesh_data[mesh_type][data_name] = load_group_recursively(data_item)
                        else:
                            mesh_data[mesh_type][data_name] = load_group_recursively(data_item)
                    elif isinstance(data_item, h5py.Dataset):
                        mesh_data[mesh_type][data_name] = load_complex_dataset(data_item)
        
        # Load other data
        metadata = load_group_recursively(f['Metadata']) if 'Metadata' in f else {}
        results_data = load_group_recursively(f['Results_Data']) if 'Results_Data' in f else {}
        
        # File information
        file_info = {
            'creation_date': f.attrs.get('creation_date', ''),
            'file_format': f.attrs.get('file_format', ''),
            'version': f.attrs.get('version', ''),
            'compression': f.attrs.get('compression', ''),
            'n_groups': f.attrs.get('n_groups', 0),
            'n_datasets': f.attrs.get('n_datasets', 0)
        }
    
    return {
        'STM': STM_data,
        'mesh_data': mesh_data,
        'metadata': metadata,
        'results_data': results_data,
        'file_info': file_info
    }


# Example usage:
if __name__ == "__main__":
    # Example of how to use the function
    print("STM Results Packaging Function")
    print("Usage:")
    print("package_STM_results(STM, mesh_data, metadata, results_data, 'output.h5')")
    print("loaded_data = load_STM_results('output.h5')")
