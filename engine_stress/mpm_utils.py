import warp as wp
from warp_utils import *

@wp.kernel
def update_x_from_u(state: MPMStateStruct):
    p = wp.tid()
    state.particle_x[p] = state.particle_x_initial[p] + state.particle_u[p]
@wp.kernel
def update_F_from_F_disp(state: MPMStateStruct):
    p = wp.tid()
    state.particle_F[p] = state.particle_F_disp[p] + wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
@wp.kernel
def update_u_from_x(state: MPMStateStruct):
    p = wp.tid()
    state.particle_u[p] = state.particle_x[p] - state.particle_x_initial[p]
@wp.kernel
def update_F_disp_from_F(state: MPMStateStruct):
    p = wp.tid()
    state.particle_F_disp[p] = state.particle_F[p] - wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
@wp.kernel
def update_F_disp_from_F_trial(state: MPMStateStruct):
    p = wp.tid()
    state.particle_F_disp[p] = state.particle_F_trial[p] - wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
@wp.kernel
def update_F_trial_from_F_disp(state: MPMStateStruct):
    p = wp.tid()
    state.particle_F_trial[p] = state.particle_F_disp[p] + wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

@wp.func
def kirchoff_stress_FCR(F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, mu: float, lam: float):
    #compute kirchoff stress for FCR model (remember tau = P F^T)
    R = U * wp.transpose(V)
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return 2.0 * mu * (F - R) * wp.transpose(F) + id * lam * J * (J - 1.0)

@wp.func
def kirchoff_stress_StVK(F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float):
    sig = wp.vec3(wp.max(sig[0], 0.01), wp.max(sig[1], 0.01), wp.max(sig[2], 0.01)) # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    log_sig_sum = wp.log(sig[0]) + wp.log(sig[1]) + wp.log(sig[2])
    ONE = wp.vec3(1.0, 1.0, 1.0)
    tau = 2.0 * mu * epsilon + lam * log_sig_sum * ONE
    return U * wp.mat33(tau[0], 0.0, 0.0, 0.0, tau[1], 0.0, 0.0, 0.0, tau[2]) * wp.transpose(V)

@wp.func
def kirchoff_stress_drucker_prager(F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float):
    log_sig_sum = wp.log(sig[0]) + wp.log(sig[1]) + wp.log(sig[2])
    center00 = 2.0 * mu * wp.log(sig[0]) * (1.0 / sig[0]) + lam * log_sig_sum * (1.0 / sig[0])
    center11 = 2.0 * mu * wp.log(sig[1]) * (1.0 / sig[1]) + lam * log_sig_sum * (1.0 / sig[1])
    center22 = 2.0 * mu * wp.log(sig[2]) * (1.0 / sig[2]) + lam * log_sig_sum * (1.0 / sig[2])
    center = wp.mat33(center00,0.0,0.0,0.0,center11,0.0,0.0,0.0,center22)
    return U * center * wp.transpose(V) * wp.transpose(F)

@wp.func
def compute_dweight(model: MPMModelStruct, w: wp.mat33, dw: wp.mat33, i: int, j: int, k: int):
    dweight = wp.vec3(dw[0,i] * w[1,j] * w[2,k],  w[0,i] * dw[1,j] * w[2,k],  w[0,i] * w[1,j] * dw[2,k])
    return dweight * model.inv_dx

@wp.func
def von_mises_return_mapping(F_trial: wp.mat33, model: MPMModelStruct, p: int):
    U = wp.mat33(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    V = wp.mat33(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)
    
    sig = wp.vec3(wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)) # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    temp = (epsilon[0]+epsilon[1]+epsilon[2])/3.0
 
    tau = 2.0 * model.mu * epsilon + model.lam * (epsilon[0]+epsilon[1]+epsilon[2]) * wp.vec3(1.0, 1.0, 1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3(tau[0] - sum_tau/3.0, tau[1] - sum_tau/3.0, tau[2] - sum_tau/3.0)
    if wp.length(cond) > model.yield_stress[p]:
        epsilon_hat = epsilon - wp.vec3(temp, temp, temp)
        epsilon_hat_norm = wp.length(epsilon_hat) + 1e-6
        delta_gamma = epsilon_hat_norm - model.yield_stress[p] / (2.0 * model.mu)
        epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        sig_elastic = wp.mat33(wp.exp(epsilon[0]), 0.0,0.0,   0.0, wp.exp(epsilon[1]), 0.0,   0.0, 0.0, wp.exp(epsilon[2]))
        F_elastic = U * sig_elastic * wp.transpose(V)
        if model.hardening == 1:
            model.yield_stress[p] = model.yield_stress[p] + 2.0 * model.mu * model.xi * delta_gamma
        return F_elastic
    else:
        return F_trial


@wp.func
def sand_return_mapping(F_trial: wp.mat33,state: MPMStateStruct, model: MPMModelStruct, p: int):
    U = wp.mat33(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    V = wp.mat33(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig, V)

    epsilon = wp.vec3(wp.log(wp.max(wp.abs(sig[0]), 1e-14)), wp.log(wp.max(wp.abs(sig[1]), 1e-14)), wp.log(wp.max(wp.abs(sig[2]), 1e-14)))
    sigma_out = wp.mat33(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0)
    tr = epsilon[0] + epsilon[1] + epsilon[2] # + state.particle_Jp[p]
    epsilon_hat = epsilon - wp.vec3(tr/3.0, tr/3.0, tr/3.0) 
    epsilon_hat_norm = wp.length(epsilon_hat) 
    delta_gamma = epsilon_hat_norm + (3.0 * model.lam + 2.0 * model.mu) / (2.0 * model.mu) * tr * model.alpha 

    if delta_gamma <= 0:
        F_elastic = F_trial
    
    if delta_gamma > 0 and tr > 0:
        F_elastic = U * wp.transpose(V)

    if delta_gamma > 0 and tr <=0:
        H = epsilon - epsilon_hat * (delta_gamma/epsilon_hat_norm)
        s_new = wp.vec3(wp.exp(H[0]), wp.exp(H[1]), wp.exp(H[2]))
       
        F_elastic = U * wp.diag(s_new) * wp.transpose(V)
    return F_elastic


@wp.kernel
def zero_grid(state: MPMStateStruct, model: MPMModelStruct): 
    grid_x, grid_y, grid_z = wp.tid()
    state.grid_m[grid_x, grid_y, grid_z] = 0.0
    state.grid_v_in[grid_x, grid_y, grid_z] = wp.vec3()
    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3()
    # for reduction
    state.neighboring_cells[grid_x, grid_y, grid_z] = 0
    state.neighboring_cells_grid_v[grid_x, grid_y, grid_z] = 0

@wp.kernel
def apply_return_mapping(state: MPMStateStruct, model: MPMModelStruct):
    # particle_F_trial -> particle_F
    p = wp.tid()
    if model.material == 1: # metal
        state.particle_F[p] = von_mises_return_mapping(state.particle_F_trial[p], model, p)
    elif model.material == 2: # sand
        state.particle_F[p] = sand_return_mapping(state.particle_F_trial[p], state, model, p)
    else: # elastic
        state.particle_F[p] = state.particle_F_trial[p]


@wp.kernel
def p2g_apic(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    p = wp.tid()
   
    J = wp.determinant(state.particle_F[p])
    U = wp.mat33(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    V = wp.mat33(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    sig = wp.vec3(0.0)

    stress = wp.mat33(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    wp.svd3(state.particle_F[p], U, sig, V)
    if model.material == 0:
        # stress = kirchoff_stress_FCR(state.particle_F[p], U, V, J, model.mu, model.lam)
        stress = kirchoff_stress_StVK(state.particle_F[p], U, V, sig, model.mu, model.lam)
    if model.material == 1:
        stress = kirchoff_stress_StVK(state.particle_F[p], U, V, sig, model.mu, model.lam)
    if model.material == 2:
        stress = kirchoff_stress_drucker_prager(state.particle_F[p], U, V, sig, model.mu, model.lam)

    state.particle_stress[p] = stress
    
    # stress = (-dt * state.particle_vol[p] * 4.0 * model.inv_dx * model.inv_dx) * stress
    # affine = stress + state.particle_mass[p] * state.particle_C[p]
    grid_pos = state.particle_x[p] * model.inv_dx
    base_pos_x = wp.int(grid_pos[0] - 0.5)
    base_pos_y =  wp.int(grid_pos[1] - 0.5)
    base_pos_z =  wp.int(grid_pos[2] - 0.5)
    fx = grid_pos - wp.vec3(wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z))
    wa = wp.vec3(1.5) - fx
    wb = fx - wp.vec3(1.0)
    wc = fx - wp.vec3(0.5)
    w = wp.mat33(
        wp.cw_mul(wa, wa) * 0.5,
        wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
        wp.cw_mul(wc, wc) * 0.5,
    )
    dw = wp.mat33(fx - wp.vec3(1.5),    -2.0*(fx - wp.vec3(1.0)),   fx - wp.vec3(0.5))

    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                dpos = (wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx) * model.dx
                ix = base_pos_x + i
                iy = base_pos_y + j
                iz = base_pos_z + k
                weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                dweight = compute_dweight(model, w, dw, i, j, k)
                # wp.atomic_add(state.grid_v_in, ix, iy, iz, weight * state.particle_mass[p] * state.particle_v[p] + weight * affine * dpos)
                wp.atomic_add(state.grid_v_in, ix, iy, iz, weight * state.particle_mass[p] * (state.particle_v[p]))#  + state.particle_C[p] * dpos))
                elastic_force = -state.particle_vol[p] * stress * dweight
                wp.atomic_add(state.grid_v_in, ix, iy, iz, dt * elastic_force)
                wp.atomic_add(state.grid_m, ix, iy, iz, weight * state.particle_mass[p])

@wp.kernel
def p2g_apic_stress_given(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    p = wp.tid()
    stress = state.particle_stress[p]
    # stress = (-dt * state.particle_vol[p] * 4.0 * model.inv_dx * model.inv_dx) * stress
    # affine = stress + state.particle_mass[p] * state.particle_C[p]
    grid_pos = state.particle_x[p] * model.inv_dx
    base_pos_x = wp.int(grid_pos[0] - 0.5)
    base_pos_y =  wp.int(grid_pos[1] - 0.5)
    base_pos_z =  wp.int(grid_pos[2] - 0.5)
    fx = grid_pos - wp.vec3(wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z))
    wa = wp.vec3(1.5) - fx
    wb = fx - wp.vec3(1.0)
    wc = fx - wp.vec3(0.5)
    w = wp.mat33(
        wp.cw_mul(wa, wa) * 0.5,
        wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
        wp.cw_mul(wc, wc) * 0.5,
    )
    dw = wp.mat33(fx - wp.vec3(1.5),    -2.0*(fx - wp.vec3(1.0)),   fx - wp.vec3(0.5))

    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                dpos = (wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx) * model.dx
                ix = base_pos_x + i
                iy = base_pos_y + j
                iz = base_pos_z + k
                weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                dweight = compute_dweight(model, w, dw, i, j, k)
                # wp.atomic_add(state.grid_v_in, ix, iy, iz, weight * state.particle_mass[p] * state.particle_v[p] + weight * affine * dpos)
                C = state.particle_C[p]
                # canshu = 0.5
                # C = (1.0 - canshu) * C + canshu/2.0 * (C-wp.transpose(C))
                wp.atomic_add(state.grid_v_in, ix, iy, iz, weight * state.particle_mass[p] * (state.particle_v[p] + C * dpos))
                elastic_force = -state.particle_vol[p] * stress * dweight
                wp.atomic_add(state.grid_v_in, ix, iy, iz, dt * elastic_force)
                wp.atomic_add(state.grid_m, ix, iy, iz, weight * state.particle_mass[p])

@wp.kernel
def grid_normalization_and_gravity(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    
    grid_x,grid_y,grid_z = wp.tid()
    m = state.grid_m[grid_x, grid_y, grid_z]
    if state.grid_m[grid_x, grid_y, grid_z] > 1e-15:
        v_out = state.grid_v_in[grid_x, grid_y, grid_z] * (1.0/ state.grid_m[grid_x, grid_y, grid_z])
        v_out = v_out + dt * wp.vec3(0.0, -1.0, 0.0) * model.gravitational_accelaration
        state.grid_v_out[grid_x, grid_y, grid_z] = v_out# wp.vec3(state.grid_v_in[grid_x, grid_y, grid_z][0]/state.grid_m[grid_x, grid_y, grid_z], state.grid_v_in[grid_x, grid_y, grid_z][1]/state.grid_m[grid_x, grid_y, grid_z], state.grid_v_in[grid_x, grid_y, grid_z][2]/state.grid_m[grid_x, grid_y, grid_z])

 

@wp.kernel
def g2p_apic_and_stress(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    p = wp.tid()
    grid_pos = state.particle_x[p] * model.inv_dx
    base_pos_x = wp.int(grid_pos[0] - 0.5)
    base_pos_y =  wp.int(grid_pos[1] - 0.5)
    base_pos_z =  wp.int(grid_pos[2] - 0.5)
    fx = grid_pos - wp.vec3(wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z))
    wa = wp.vec3(1.5) - fx
    wb = fx - wp.vec3(1.0)
    wc = fx - wp.vec3(0.5)
    w = wp.mat33(
        wp.cw_mul(wa, wa) * 0.5,
        wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
        wp.cw_mul(wc, wc) * 0.5,
    )
    dw = wp.mat33(fx - wp.vec3(1.5),    -2.0*(fx - wp.vec3(1.0)),   fx - wp.vec3(0.5))
    new_v = wp.vec3(0.0,0.0,0.0)
    new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    new_F = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                ix = base_pos_x + i
                iy = base_pos_y + j
                iz = base_pos_z + k
                dpos = (wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx)
                weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                grid_v = state.grid_v_out[ix,iy,iz]
                new_v = new_v + grid_v * weight
                new_C = new_C + wp.outer(grid_v, dpos) * (weight * model.inv_dx * 4.0)
                dweight = compute_dweight(model, w, dw, i, j, k)
                new_F = new_F + wp.outer(grid_v, dweight)


    state.particle_v[p] = new_v
    state.particle_x[p] = state.particle_x[p] + dt * new_v
    state.particle_C[p] = new_C
    I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    # F_tmp = (I33 + state.particle_C[p] * dt) * state.particle_F[p]
    F_tmp = (I33 + new_F * dt) * state.particle_F[p]
    state.particle_F_trial[p] = F_tmp

    if model.material == 1: # metal
        state.particle_F[p] = von_mises_return_mapping(state.particle_F_trial[p], model, p)
    elif model.material == 2: # sand
        state.particle_F[p] = sand_return_mapping(state.particle_F_trial[p], state, model, p)
    else: # elastic
        state.particle_F[p] = state.particle_F_trial[p]

    # also compute stress here
    J = wp.determinant(state.particle_F[p])
    U = wp.mat33(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    V = wp.mat33(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    sig = wp.vec3(0.0)
    stress = wp.mat33(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    wp.svd3(state.particle_F[p], U, sig, V)
    if model.material == 0:
        stress = kirchoff_stress_FCR(state.particle_F[p], U, V, J, model.mu, model.lam)
        # stress = kirchoff_stress_StVK(state.particle_F[p], U, V, sig, model.mu, model.lam)
    if model.material == 1:
        stress = kirchoff_stress_StVK(state.particle_F[p], U, V, sig, model.mu, model.lam)
    if model.material == 2:
        stress = kirchoff_stress_drucker_prager(state.particle_F[p], U, V, sig, model.mu, model.lam)

    state.particle_stress[p] = stress





#-----------------------------------------------------
#-----------------------------------------------------
#-----------------------------------------------------
# below are for reduction only


@wp.kernel
def update_neighboring_cells(state: MPMStateStruct, model: MPMModelStruct):
    # loop over sampling points
    p = wp.tid()
    sample_point_index = state.sample_particle_index_short[p]
    grid_pos = state.particle_x[sample_point_index] * model.inv_dx
    base_pos_x = wp.int(grid_pos[0] - 0.5)
    base_pos_y = wp.int(grid_pos[1] - 0.5)
    base_pos_z = wp.int(grid_pos[2] - 0.5)

    for i in range(-2,3):
        for j in range(-2,3):
            for k in range(-2,3):
                # -2, -1, 0, 1, 2
                if 0<=base_pos_x+i and base_pos_x+i<model.grid_dim_x and 0<=base_pos_y+j and base_pos_y+j<model.grid_dim_y and 0<=base_pos_z+k and base_pos_z+k<model.grid_dim_z:
                    state.neighboring_cells[base_pos_x+i, base_pos_y+j, base_pos_z+k] = 1
                    # if state.neighboring_cells[base_pos_x+i, base_pos_y+j, base_pos_z+k] == 0:
                    # wp.atomic_add(state.neighboring_cells, base_pos_x+i, base_pos_y+j, base_pos_z+k, 1)

@wp.kernel
def update_neighboring_cells_for_grid_v(state: MPMStateStruct, model: MPMModelStruct):
    # loop over sampling points
    p = wp.tid()
    sample_point_index = state.sample_particle_index_short[p]
    grid_pos = state.particle_x[sample_point_index] * model.inv_dx
    base_pos_x = wp.int(grid_pos[0] - 0.5)
    base_pos_y = wp.int(grid_pos[1] - 0.5)
    base_pos_z = wp.int(grid_pos[2] - 0.5)

    for i in range(-2,5):
        for j in range(-2,5):
            for k in range(-2,5):
                # -2, -1, 0, 1, 2, 3, 4
                if 0<=base_pos_x+i and base_pos_x+i<model.grid_dim_x and 0<=base_pos_y+j and base_pos_y+j<model.grid_dim_y and 0<=base_pos_z+k and base_pos_z+k<model.grid_dim_z:
                    state.neighboring_cells_grid_v[base_pos_x+i, base_pos_y+j, base_pos_z+k] = 1
                    # if state.neighboring_cells[base_pos_x+i, base_pos_y+j, base_pos_z+k] == 0:
                    # wp.atomic_add(state.neighboring_cells, base_pos_x+i, base_pos_y+j, base_pos_z+k, 1)

@wp.kernel
def p2g_sample_all(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    p = wp.tid()
    grid_pos = state.particle_x[p] * model.inv_dx
    base_pos_x = wp.int(grid_pos[0] - 0.5)
    base_pos_y = wp.int(grid_pos[1] - 0.5)
    base_pos_z = wp.int(grid_pos[2] - 0.5)
    if state.neighboring_cells[base_pos_x, base_pos_y, base_pos_z]>0:
        # only update necessary F
        state.particle_F[p] = state.particle_F_disp[p] + wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        U = wp.mat33(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
        V = wp.mat33(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
        sig = wp.vec3(0.0)

        wp.svd3(state.particle_F[p], U, sig, V)
        J = 1.0
        for d in range(3):
            J = J * sig[d]
        stress = wp.mat33(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
        if model.material == 0 or model.material == 1: 
            # Compute Kirchoff Stress
            stress = kirchoff_stress_FCR(state.particle_F[p], U, V, J, model.mu, model.lam)
        elif model.material == 2:
            stress = kirchoff_stress_drucker_prager(state.particle_F[p], U, V, sig, model.mu, model.lam)

        fx = grid_pos - wp.vec3(wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z))
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        dw = wp.mat33(fx - wp.vec3(1.5),    -2.0*(fx - wp.vec3(1.0)),   fx - wp.vec3(0.5))

        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k

                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                
                    wp.atomic_add(state.grid_v_in, ix, iy, iz, weight * state.particle_mass[p] * state.particle_v[p])
                    wp.atomic_add(state.grid_m, ix, iy, iz, weight * state.particle_mass[p])

                    dweight = compute_dweight(model, w, dw, i, j, k)
                    elastic_force = - (state.particle_mass[p]/state.particle_density[p]) * stress * dweight 
                    body_force = wp.vec3(0.0,0.0,0.0)
                    force = elastic_force + body_force
                
                    wp.atomic_add(state.grid_v_in, ix, iy, iz, dt * force) # add elastic force to update velocity, don't divide by mass bc this is actually updating MOMENTUM


@wp.kernel
def grid_normalization_and_gravity_sample_all(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    grid_x, grid_y, grid_z = wp.tid()
    if state.neighboring_cells[grid_x, grid_y, grid_z]>0:
        m = state.grid_m[grid_x, grid_y, grid_z]
        if m > 1e-10:
            v_out = state.grid_v_in[grid_x, grid_y, grid_z] / state.grid_m[grid_x, grid_y, grid_z]
            v_out = v_out + dt * wp.vec3(0.0, -1.0, 0.0) * model.gravitational_accelaration
            state.grid_v_out[grid_x, grid_y, grid_z] = v_out

@wp.kernel
def g2p_sample(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    p = wp.tid()
    sample_point_index = state.sample_particle_index_short[p] # this should be an uint32, particle index
    grid_pos = state.particle_x[sample_point_index] * model.inv_dx

    base_pos_x = wp.int(grid_pos[0] - 0.5)
    base_pos_y = wp.int(grid_pos[1] - 0.5)
    base_pos_z = wp.int(grid_pos[2] - 0.5)
    fx = grid_pos - wp.vec3(wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z))
    wa = wp.vec3(1.5) - fx
    wb = fx - wp.vec3(1.0)
    wc = fx - wp.vec3(0.5)
    w = wp.mat33(
        wp.cw_mul(wa, wa) * 0.5,
        wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
        wp.cw_mul(wc, wc) * 0.5,
    )
    dw = wp.mat33(fx - wp.vec3(1.5),    -2.0*(fx - wp.vec3(1.0)),   fx - wp.vec3(0.5))

    new_v = wp.vec3(0.0,0.0,0.0)
    new_F = wp.mat33(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                ix = base_pos_x + i
                iy = base_pos_y + j
                iz = base_pos_z + k

                weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                grid_v = state.grid_v_out[ix,iy,iz]
                new_v = new_v + grid_v * weight

                dweight = compute_dweight(model, w, dw, i, j, k)
                new_F = new_F + wp.outer(grid_v, dweight)

    state.particle_v[sample_point_index] = new_v
    state.particle_x[sample_point_index] = state.particle_x[sample_point_index] + dt * new_v
    F_trial = (wp.mat33(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0) + dt * new_F) * state.particle_F[sample_point_index]

    if model.material == 1:
        state.particle_F[sample_point_index] = von_mises_return_mapping(F_trial, model, p)
    elif model.material == 2:
        state.particle_F[sample_point_index] = sand_return_mapping(F_trial, state, model, p)
    else:
        state.particle_F[sample_point_index] = F_trial
    # originally ufromx and F_disp from F
    state.particle_u[sample_point_index] = state.particle_x[sample_point_index] - state.particle_x_initial[sample_point_index]
    state.particle_F_disp[sample_point_index] = state.particle_F[sample_point_index] - wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
