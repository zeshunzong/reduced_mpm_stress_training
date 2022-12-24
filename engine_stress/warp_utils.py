import warp as wp
import warp.torch
import torch

def torch2warp_vec3(t, dtype=warp.types.float32, dvc = "cuda:0"):
    assert(t.is_contiguous())

    if (t.dtype != torch.float32 and t.dtype != torch.int32):
        raise RuntimeError("Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type")

    assert(t.shape[1]==3)

    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.vec3,
        shape=t.shape[0],
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc)
    a.tensor = t
    return a

def torch2warp_mat33(t, dtype=warp.types.float32, dvc = "cuda:0"):
    assert(t.is_contiguous())

    if (t.dtype != torch.float32 and t.dtype != torch.int32):
        raise RuntimeError("Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type")

    assert(t.shape[1]==3)

    a = warp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.mat33,
        shape=t.shape[0],
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc)
    a.tensor = t
    return a
def torch2warp_gridv(t, dtype=wp.types.float32, dvc = "cuda:0"):
    assert(t.is_contiguous())
    if (t.dtype != torch.float32 and t.dtype != torch.int32):
        raise RuntimeError("Error aliasing Torch tensor to Warp array. Torch tensor must be float32 or int32 type")
    assert(t.shape[3]==3)
    a = wp.types.array(
        ptr=t.data_ptr(),
        dtype=wp.vec3,
        shape=(t.shape[0], t.shape[1], t.shape[2]),
        copy=False,
        owner=False,
        requires_grad=t.requires_grad,
        # device=t.device.type)
        device=dvc)
    a.tensor = t
    return a

@wp.struct
class MPMModelStruct:
    grid_lim: float
    n_particles: int
    n_grid: int
    dx: float
    inv_dx: float
    grid_dim_x: int
    grid_dim_y: int
    grid_dim_z: int
    mu: float
    lam: float
    E: float
    nu: float
    yield_stress: wp.array(dtype=float)
    material: int
    friction_angle: float
    alpha: float
    gravitational_accelaration: float
    hardening: int
    xi: float
    

@wp.struct
class MPMStateStruct:
    n_grid: int
    dx: float
    inv_dx: float
    # grid_lower: wp.array(dtype=int)
    # grid_upper: wp.array(dtype=int)

    particle_x_initial: wp.array(dtype=wp.vec3) # initial position
    particle_x: wp.array(dtype=wp.vec3) # current position
    particle_u: wp.array(dtype=wp.vec3) # displacement = current position - initial position
    particle_v: wp.array(dtype=wp.vec3) # particle velocity
    particle_F: wp.array(dtype=wp.mat33) # particle elastic deformation gradient
    particle_F_trial: wp.array(dtype=wp.mat33) # apply return mapping on this will yield elastic def grad
    particle_stress:  wp.array(dtype=wp.mat33) # Kirchoff stress
    particle_F_disp: wp.array(dtype=wp.mat33) # particle elastic deformation gradient 
    particle_C: wp.array(dtype=wp.mat33)

    particle_Jp: wp.array(dtype=float)
   
    particle_vol: wp.array(dtype=float)  # current volume
    particle_mass: wp.array(dtype=float) # mass
    particle_density: wp.array(dtype=float) # density
    particle_external_force: wp.array(dtype=wp.vec3) # external force excerted on particle

    grid_m: wp.array(dtype=float, ndim=3)
    # grid_mv: wp.array(dtype=wp.vec3, ndim=3)
    grid_v_in: wp.array(dtype=wp.vec3, ndim=3) # grid node momentum/velocity
    grid_v_out: wp.array(dtype=wp.vec3, ndim=3) # grid node momentum/velocity, after grid update
    # error: wp.array(dtype=int)

    # particle_color: wp.array(dtype=float)  # color for obj output

    neighboring_cells: wp.array(dtype = int, ndim = 3)
    neighboring_cells_grid_v: wp.array(dtype = int, ndim = 3)

    sample_particle_index_short: wp.array(dtype = int, ndim = 1)
    sample_particle_index_long:  wp.array(dtype = int, ndim = 1)
    sample_particle_all_index: wp.array(dtype = int, ndim = 1) # length = n_particles, 1 at quadratures, 0 o.w.
    quadrature_indices_short: wp.array(dtype = int, ndim = 1) # length = num of quadratures, indices
    particle_color_temp: wp.array(dtype = int, ndim = 1)

@wp.struct
class Dirichlet_collider:

    point: wp.vec3
    normal: wp.vec3

    start_time: float
    end_time: float


    friction: float
    surface_type: int

    velocity: wp.vec3

    threshold: float
    reset: int

    x_unit: wp.vec3
    y_unit: wp.vec3
    radius: float
    v_scale: float
    width: float
    height: float
    R: float
    width: float
    
@wp.struct
class MPMtailoredStruct:
    # this needs to be changed for each different BC!
    point: wp.vec3
    normal: wp.vec3
    start_time: float
    end_time: float
    friction: float
    surface_type: int
    velocity: wp.vec3
    threshold: float
    reset: int

    point_rotate: wp.vec3
    normal_rotate: wp.vec3
    x_unit: wp.vec3
    y_unit: wp.vec3
    radius: float
    v_scale: float
    width: float
    point_plane: wp.vec3
    normal_plane: wp.vec3
    velocity_plane: wp.vec3
    threshold_plane: float


@wp.kernel
def set_vec3_to_zero(target_array: wp.array(dtype = wp.vec3)):
    tid = wp.tid()
    target_array[tid] = wp.vec3(0.0, 0.0, 0.0)

@wp.kernel
def set_mat33_to_identity(target_array: wp.array(dtype = wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

@wp.kernel
def add_identity_to_mat33(target_array: wp.array(dtype = wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.add(target_array[tid], wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

@wp.kernel
def subtract_identity_to_mat33(target_array: wp.array(dtype = wp.mat33)):
    tid = wp.tid()
    target_array[tid] = wp.sub(target_array[tid], wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

@wp.kernel
def add_vec3_to_vec3(first_array: wp.array(dtype = wp.vec3), second_array: wp.array(dtype = wp.vec3)):
    tid = wp.tid()
    first_array[tid] = wp.add(first_array[tid], second_array[tid])

@wp.kernel
def set_value_to_float_array(target_array: wp.array(dtype = float), value: float):
    tid = wp.tid()
    target_array[tid] = value

@wp.kernel
def get_float_array_product(arrayA: wp.array(dtype = float),arrayB: wp.array(dtype = float),arrayC: wp.array(dtype = float)):
    tid = wp.tid()
    arrayC[tid] = arrayA[tid] * arrayB[tid]

