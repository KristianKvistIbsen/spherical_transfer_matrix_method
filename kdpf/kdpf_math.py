from ansys.dpf import core as dpf
from kdpf.kdpf_mesh import get_normals
import numpy as np

def compute_flux_through_skin_mesh(flux_vector_field,skin_mesh, location="Nodal", asOperator=False):
    
    skin_mesh_scoping = skin_mesh.elements.scoping #get_scoping_from_mesh(skin_mesh,"Elemental")

    op = dpf.operators.mapping.solid_to_skin() # operator instantiation
    op.inputs.field.connect(flux_vector_field)
    op.inputs.mesh.connect(skin_mesh)
    op.inputs.solid_mesh.connect(flux_vector_field.meshed_region)# optional
    interface_flux = op.outputs.field()

    normals = get_normals(skin_mesh)

    op = dpf.operators.math.generalized_inner_product() # operator instantiation
    op.inputs.fieldA.connect(interface_flux)
    op.inputs.fieldB.connect(normals)
    normal_interface_flux = op.outputs.field()

    op_integrator = dpf.operators.geo.integrate_over_elements()
    op_integrator.inputs.field.connect(normal_interface_flux)
    op_integrator.inputs.scoping.connect(skin_mesh_scoping)
    op_integrator.inputs.mesh.connect(skin_mesh)
    integrated_normal_interface_flux = op_integrator.outputs.field()
    integrated_normal_interface_flux.meshed_region = skin_mesh

    if location == "Elemental":
        if asOperator:
            return op_integrator
        else:    
            return integrated_normal_interface_flux
    else:
        op = dpf.operators.averaging.to_nodal()
        op.inputs.field.connect(integrated_normal_interface_flux)
        nodal_field = op.outputs.field()
        if asOperator:
            return op
        else:    
            return nodal_field

def scale_field_container(fc, factor, asOperator=False):
    op_scale = dpf.operators.math.scale_fc()
    op_scale.inputs.fields_container.connect(fc)
    op_scale.inputs.ponderation.connect(float(factor))
    if asOperator:
        return op_scale
    else:
        return op_scale.outputs.fields_container()

def scale_field(field, factor, asOperator=False):
    op_scale = dpf.operators.math.scale()
    op_scale.inputs.field.connect(field)
    op_scale.inputs.ponderation.connect(float(factor))
    if asOperator:
        return op_scale
    else:
        return op_scale.outputs.field()

def multiply_complex_fields(field1, field2, asOperator=False):
    op_product = dpf.operators.math.cplx_multiply()
    op_product.inputs.fields_containerA.connect(field1)
    op_product.inputs.fields_containerB.connect(field2)
    if asOperator:
        return op_product
    else:
        return op_product.outputs.fields_container()

def add_fields(field1, field2, asOperator=False):
    op_add = dpf.operators.math.add_fc()
    op_add.inputs.fields_container1.connect(field1)
    op_add.inputs.fields_container2.connect(field2)
    if asOperator:
        return op_add
    else:
        return op_add.outputs.fields_container()
    
def subtract_field_containers(field1,field2):  
    op_subtract = dpf.operators.math.minus_fc()
    op_subtract.inputs.field_or_fields_container_A.connect(field1)
    op_subtract.inputs.field_or_fields_container_B.connect(field2)
    return op_subtract.outputs.fields_container()

def conjugate_field_container(field_container):
    op_conj = dpf.operators.math.conjugate()
    op_conj.inputs.fields_container.connect(field_container)
    return op_conj.outputs.fields_container()

def integrate(field,scoping=None):
    if scoping == None:
        scoping = field.meshed_region.elements.scoping #get_scoping_from_mesh(field.meshed_region, "Elemental")
    op_integral = dpf.operators.geo.integrate_over_elements() # operator instantiation
    op_integral.inputs.field.connect(field)
    op_integral.inputs.scoping.connect(scoping)# optional
    op_integral.inputs.mesh.connect(field.meshed_region)# optional

    return op_integral.outputs.field()

def dot_fields(field1,field2):
    opDot = dpf.operators.math.generalized_inner_product_fc()
    opDot.inputs.field_or_fields_container_A.connect(field1)
    opDot.inputs.field_or_fields_container_B.connect(field2)
    return opDot.outputs.fields_container()

def compute_strain_energy_density_damped_harmonic(model, mesh, tfreq):

    op_strain = dpf.operators.result.elastic_strain() # operator instantiation
    op_strain.inputs.time_scoping.connect(tfreq)# optional
    op_strain.inputs.data_sources.connect(model)
    op_strain.inputs.mesh.connect(mesh)# optional
    op_strain.inputs.requested_location.connect("Nodal")# optional
    eps = op_strain.outputs.fields_container()

    # Extract strain tensor components
    epsx = eps.select_component(0)
    epsy = eps.select_component(1)
    epsz = eps.select_component(2)
    epsxy = eps.select_component(3)
    epsyz = eps.select_component(4)
    epsxz = eps.select_component(5)

    op_stress = dpf.operators.result.stress() # operator instantiation
    op_stress.inputs.time_scoping.connect(tfreq)# optional
    op_stress.inputs.data_sources.connect(model)
    op_stress.inputs.mesh.connect(mesh)# optional
    op_stress.inputs.requested_location.connect("Nodal")# optional
    sig = op_stress.outputs.fields_container()

    # Extract stress tensor components
    sigx = sig.select_component(0)
    sigy = sig.select_component(1)
    sigz = sig.select_component(2)
    sigxy = sig.select_component(3)
    sigyz = sig.select_component(4)
    sigxz = sig.select_component(5)

    sxex = multiply_complex_fields(sigx, epsx)
    syey = multiply_complex_fields(sigy, epsy)
    szez = multiply_complex_fields(sigz, epsz)
    sxyexy = multiply_complex_fields(sigxy, epsxy) # I THINK MAYBE THESE SHOULD NOT BE MULTIPLIED WITH 2!!!
    sxzexz = multiply_complex_fields(sigxz, epsxz)
    syzeyz = multiply_complex_fields(sigyz, epsyz)

    # sxyexy = scale_field(multiply_fields(sigxy, epsxy),2) # I THINK MAYBE THESE SHOULD NOT BE MULTIPLIED WITH 2!!!
    # sxzexz = scale_field(multiply_fields(sigxz, epsxz),2)
    # syzeyz = scale_field(multiply_fields(sigyz, epsyz),2)

    print("Warning in compute_strain_energy_density_damped_harmonic: Maybe cross terms must be x2")

    sxex_syey = add_fields(sxex,syey)
    sxex_syey_szez = add_fields(sxex_syey,szez)
    sxex_syey_szez_sxyexy = add_fields(sxex_syey_szez,sxyexy)
    sxex_syey_szez_sxyexy_sxzexz = add_fields(sxex_syey_szez_sxyexy,sxzexz)
    sxex_syey_szez_sxyexy_sxzexz_syzeyz = add_fields(sxex_syey_szez_sxyexy_sxzexz,syzeyz)

    return scale_field_container(sxex_syey_szez_sxyexy_sxzexz_syzeyz,1/2)


def compute_complex_tensor_product(T1, T2, scale_factor=1.0):
    """
    Perform tensor product between ij and j tensors of complex values and rescale using scale factor,
    including component extraction and vectorization.

    Parameters:
    T1 (DPF FieldsContainer): e.g. Stress tensor fields container of 6 fields
    T2 (DPF FieldsContainer): Conjugate velocity vector fields container
    scale_factor (float): Scale factor to apply to the final result (default: 1)

    Returns:
    DPF Field: Vectorized and scaled result
    """
    op_vectorize = dpf.operators.utility.assemble_scalars_to_vectors()

    # Extract stress tensor components
    T1x = T1.select_component(0)
    T1y = T1.select_component(1)
    T1z = T1.select_component(2)
    T1xy = T1.select_component(3)
    T1yz = T1.select_component(4)
    T1xz = T1.select_component(5)

    # Extract velocity vector components
    T2x = T2.select_component(0)
    T2y = T2.select_component(1)
    T2z = T2.select_component(2)

    # Calculate x component: T1x*T2x + T1xy*T2y + T1xz*T2z
    x_comp1 = multiply_complex_fields(T1x, T2x)
    x_comp2 = multiply_complex_fields(T1xy, T2y)
    x_comp3 = multiply_complex_fields(T1xz, T2z)
    x_partial = add_fields(x_comp1, x_comp2)
    x_component = add_fields(x_partial, x_comp3)

    # Calculate y component: T1xy*T2x + T1y*T2y + T1yz*T2z
    y_comp1 = multiply_complex_fields(T1xy, T2x)
    y_comp2 = multiply_complex_fields(T1y, T2y)
    y_comp3 = multiply_complex_fields(T1yz, T2z)
    y_partial = add_fields(y_comp1, y_comp2)
    y_component = add_fields(y_partial, y_comp3)

    # Calculate z component: T1xz*T2x + T1yz*T2y + T1z*T2z
    z_comp1 = multiply_complex_fields(T1xz, T2x)
    z_comp2 = multiply_complex_fields(T1yz, T2y)
    z_comp3 = multiply_complex_fields(T1z, T2z)
    z_partial = add_fields(z_comp1, z_comp2)
    z_component = add_fields(z_partial, z_comp3)

    # Vectorize the real result
    op_vectorize.inputs.x.connect(x_component.get_fields({"complex":0})[0])
    op_vectorize.inputs.y.connect(y_component.get_fields({"complex":0})[0])
    op_vectorize.inputs.z.connect(z_component.get_fields({"complex":0})[0])
    T1T2_real_unscaled = op_vectorize.outputs.field()

    # Scale the results
    op_scale = dpf.operators.math.scale()
    op_scale.inputs.field.connect(T1T2_real_unscaled)
    op_scale.inputs.ponderation.connect(scale_factor)
    T1T2_real = op_scale.outputs.field()

    # Vectorize the real result
    op_vectorize.inputs.x.connect(x_component.get_fields({"complex":1})[0])
    op_vectorize.inputs.y.connect(y_component.get_fields({"complex":1})[0])
    op_vectorize.inputs.z.connect(z_component.get_fields({"complex":1})[0])
    T1T2_imag_unscaled = op_vectorize.outputs.field()

    # Scale the results
    op_scale = dpf.operators.math.scale()
    op_scale.inputs.field.connect(T1T2_imag_unscaled)
    op_scale.inputs.ponderation.connect(scale_factor)
    T1T2_imag = op_scale.outputs.field()

    return T1T2_real, T1T2_imag


def compute_erp(model,skin_mesh,tfreq,skin_mesh_scoping,RHO,C,REFERENCE_POWER):
    op_displacement = dpf.operators.result.displacement()
    op_displacement.inputs.time_scoping.connect(tfreq)
    op_displacement.inputs.mesh_scoping.connect(skin_mesh_scoping)
    op_displacement.inputs.data_sources.connect(model)
    op_displacement.inputs.mesh.connect(skin_mesh)
    displacements_fc = op_displacement.outputs.fields_container()

    # Perform the dpf ERP calculation
    op_erp = dpf.operators.result.equivalent_radiated_power()
    op_erp.inputs.fields_container.connect(displacements_fc)
    op_erp.inputs.mesh.connect(skin_mesh)
    op_erp.inputs.mass_density.connect(RHO)
    op_erp.inputs.speed_of_sound.connect(C)
    op_erp.inputs.erp_type.connect(0)
    op_erp.inputs.factor.connect(REFERENCE_POWER)
    erp_fc = op_erp.outputs.fields_container()
    return erp_fc

def multiply_fields_container_by_complex(fc, complex_value, scoping=None): #WITH UNTESTED SCOPING
    """
    Multiply a fields container representing complex fields by a complex scalar.

    The fields container is expected to have labels 'time' and 'complex',
    where 'complex' is 0 for real parts and 1 for imaginary parts, paired by time steps.
    If a scoping is provided, only the entities specified in the scoping are scaled;
    otherwise, all entities are scaled.

    Args:
        fc (ansys.dpf.core.FieldsContainer): The input fields container with 'time' and 'complex' labels.
        complex_value (complex): The complex scalar (a + b*j) to multiply by.
        scoping (ansys.dpf.core.Scoping, optional): The scoping to specify which entities to scale.
            If None, all entities are scaled.

    Returns:
        ansys.dpf.core.FieldsContainer: A new fields container with the multiplied fields.
    """
    # Extract real and imaginary parts of the complex scalar
    a = complex_value.real
    b = complex_value.imag

    # Get all unique time steps from the fields container
    time_steps = fc.get_available_ids_for_label("time")

    # Initialize a new fields container with the same labels
    new_fc = dpf.FieldsContainer()
    new_fc.labels = ["time", "complex"]

    # Process each time step
    for t in time_steps:
        # Extract the real and imaginary fields for the current time step
        field_real = fc.get_fields({"time": t, "complex": 0})[0]
        field_imag = fc.get_fields({"time": t, "complex": 1})[0]

        # Get the scoping and number of components from the real field
        # (assuming real and imaginary fields have the same scoping and component count)
        scoping_field = field_real.scoping
        num_components = field_real.component_count

        # Reshape data to (number of entities, number of components) for easier manipulation
        data_real = field_real.data.reshape(-1, num_components)
        data_imag = field_imag.data.reshape(-1, num_components)

        # Determine which indices to scale based on the scoping
        field_ids = np.array(scoping_field.ids)
        if scoping is None:
            scoping = field_real.scoping

        # Scale only entities in the provided scoping
        scoping_ids = np.array(scoping.ids)
        indices_to_scale = np.where(np.isin(field_ids, scoping_ids))[0]

        # Create copies of the original data
        new_data_real = data_real.copy()
        new_data_imag = data_imag.copy()

        # Apply complex multiplication only to the specified indices
        if len(indices_to_scale) > 0:
            new_data_real[indices_to_scale, :] = (
                a * data_real[indices_to_scale, :] - b * data_imag[indices_to_scale, :]
            )
            new_data_imag[indices_to_scale, :] = (
                b * data_real[indices_to_scale, :] + a * data_imag[indices_to_scale, :]
            )

        # Create new fields by cloning the original fields and updating their data
        new_field_real = field_real.deep_copy()
        new_field_real.data = new_data_real.flatten()

        new_field_imag = field_imag.deep_copy()
        new_field_imag.data = new_data_imag.flatten()

        # Add the new fields to the output fields container with correct labels
        new_fc.add_field({"time": t, "complex": 0}, new_field_real)
        new_fc.add_field({"time": t, "complex": 1}, new_field_imag)

    return new_fc

def multiply_fields_container_by_complex_array(fc, complex_array, scoping=None):
    """
    Multiply a fields container representing complex fields by an array of complex scalars.

    The fields container is expected to have labels 'time' and 'complex',
    where 'complex' is 0 for real parts and 1 for imaginary parts, paired by time steps.
    The complex_array must have a length equal to the number of entities in the field's scoping.
    If a scoping is provided, only the entities specified in the scoping are scaled;
    otherwise, all entities are scaled.

    Args:
        fc (ansys.dpf.core.FieldsContainer): The input fields container with 'time' and 'complex' labels.
        complex_array (np.ndarray): Array of complex scalars, one for each entity in the field's scoping.
        scoping (ansys.dpf.core.Scoping, optional): The scoping to specify which entities to scale.
            If None, all entities are scaled.

    Returns:
        ansys.dpf.core.FieldsContainer: A new fields container with the multiplied fields.

    Raises:
        ValueError: If the length of complex_array does not match the number of entities.
    """
    # Ensure complex_array is a numpy array
    if not isinstance(complex_array, np.ndarray):
        complex_array = np.array(complex_array, dtype=complex)

    # Get the scoping from the first field to determine the number of entities
    time_steps = fc.get_available_ids_for_label("time")
    field_real_first = fc.get_fields({"time": time_steps[0], "complex": 0})[0]
    scoping_field = field_real_first.scoping
    num_entities = len(scoping_field.ids)

    # Check that complex_array length matches the number of entities
    if len(complex_array) != num_entities:
        raise ValueError(f"complex_array must have length {num_entities}, got {len(complex_array)}")

    # Initialize a new fields container with the same labels
    new_fc = dpf.FieldsContainer()
    new_fc.labels = ["time", "complex"]

    # Process each time step
    for t in time_steps:
        # Extract the real and imaginary fields for the current time step
        field_real = fc.get_fields({"time": t, "complex": 0})[0]
        field_imag = fc.get_fields({"time": t, "complex": 1})[0]

        # Get the number of components from the real field
        num_components = field_real.component_count

        # Reshape data to (number of entities, number of components)
        data_real = field_real.data.reshape(-1, num_components)
        data_imag = field_imag.data.reshape(-1, num_components)

        # Determine which indices to scale based on the scoping
        field_ids = np.array(scoping_field.ids)
        if scoping is None:
            indices_to_scale = np.arange(num_entities)
        else:
            scoping_ids = np.array(scoping.ids)
            indices_to_scale = np.where(np.isin(field_ids, scoping_ids))[0]

        # Create copies of the original data
        new_data_real = data_real.copy()
        new_data_imag = data_imag.copy()

        # Apply complex multiplication only to the specified indices
        if len(indices_to_scale) > 0:
            # Extract real and imaginary parts of the complex array for the indices to scale
            a_scale = complex_array[indices_to_scale].real
            b_scale = complex_array[indices_to_scale].imag

            # Perform complex multiplication using broadcasting
            new_data_real[indices_to_scale, :] = (
                a_scale[:, None] * data_real[indices_to_scale, :] - b_scale[:, None] * data_imag[indices_to_scale, :]
            )
            new_data_imag[indices_to_scale, :] = (
                b_scale[:, None] * data_real[indices_to_scale, :] + a_scale[:, None] * data_imag[indices_to_scale, :]
            )

        # Create new fields by cloning the original fields and updating their data
        new_field_real = field_real.deep_copy()
        new_field_real.data = new_data_real.flatten()

        new_field_imag = field_imag.deep_copy()
        new_field_imag.data = new_data_imag.flatten()

        # Add the new fields to the output fields container with correct labels
        new_fc.add_field({"time": t, "complex": 0}, new_field_real)
        new_fc.add_field({"time": t, "complex": 1}, new_field_imag)

    return new_fc

def compute_complex_field_norm(fc):

    # Extract the real and imaginary fields based on the 'complex' label
    field_real = None
    field_imag = None
    for i in range(len(fc)):
        label_space = fc.get_label_space(i)
        if label_space.get("complex", None) == 0:
            field_real = fc.get_field(i)
        elif label_space.get("complex", None) == 1:
            field_imag = fc.get_field(i)

    # Verify that both fields were found
    if field_real is None or field_imag is None:
        raise ValueError("Fields container must contain fields with 'complex' labels 0 and 1.")

    # Compute the square of the real part
    op_sqr_real = dpf.operators.math.sqr()
    op_sqr_real.inputs.field.connect(field_real)
    field_sqr_real = op_sqr_real.outputs.field()

    # Compute the square of the imaginary part
    op_sqr_imag = dpf.operators.math.sqr()
    op_sqr_imag.inputs.field.connect(field_imag)
    field_sqr_imag = op_sqr_imag.outputs.field()

    # Add the squared fields
    op_add = dpf.operators.math.add()
    op_add.inputs.fieldA.connect(field_sqr_real)
    op_add.inputs.fieldB.connect(field_sqr_imag)
    field_sum_sqrs = op_add.outputs.field()

    # Compute the square root to obtain the magnitude
    op_sqrt = dpf.operators.math.sqrt()
    op_sqrt.inputs.field.connect(field_sum_sqrs)
    magnitude_field = op_sqrt.outputs.field()

    return magnitude_field

def field_from_array_vector(array, mesh, num_components):
    """
    Create a vector field from a 1D array by replicating each scalar across all components.

    Args:
        array (np.ndarray): 1D array of scalars, one per node.
        mesh (ansys.dpf.core.MeshedRegion): The mesh to associate with the field.
        num_components (int): Number of components in the vector field (e.g., 3 for x, y, z).

    Returns:
        ansys.dpf.core.Field: A vector field with the array values replicated across components.
    """
    n_nodes = mesh.nodes.n_nodes
    if len(array) != n_nodes:
        raise ValueError(f"Array length {len(array)} does not match number of nodes {n_nodes}")

    # Create a vector field
    field = dpf.fields_factory.create_3d_vector_field(n_nodes, dpf.locations.nodal)
    field.scoping = mesh.nodes.scoping
    field.meshed_region = mesh

    # Replicate the scalar array across all components
    data = np.repeat(array[:, np.newaxis], num_components, axis=1)  # Shape: [n_nodes, num_components]
    field.data = data.flatten()  # DPF expects flattened data

    return field