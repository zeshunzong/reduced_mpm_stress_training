import warp as wp
import sys
import warp.torch
from timer_cm import Timer
import numpy as np
import argparse
import h5py
import torch
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from mpm_utils import *
from warp_utils import * 

def flatten2nD(index, shape_tuple):
    index_tuple = [0] * len(shape_tuple)
    for dim in range(1, len(shape_tuple)+1):
        multiplier = 1
        for k in range(dim, len(shape_tuple)):
            multiplier = multiplier * shape_tuple[k]
        index_tuple[dim-1] = int(index/multiplier)
        index = index % multiplier
    return index_tuple

class MPM_Simulator_WARP:
    def __init__(self, n_particles, n_grid = 100, grid_lim = 1.0, device = "cuda:0"):
        self.initialize(n_particles, n_grid, grid_lim, device = device)

    def initialize(self, n_particles, n_grid = 100, grid_lim = 1.0, device = "cuda:0"): 
        self.n_particles = n_particles

        self.mpm_model = MPMModelStruct()
        self.mpm_model.grid_lim = grid_lim # domain will be [0,grid_lim]*[0,grid_lim]*[0,grid_lim]
        self.mpm_model.n_grid = n_grid
        self.mpm_model.grid_dim_x = self.mpm_model.n_grid
        self.mpm_model.grid_dim_y = self.mpm_model.n_grid
        self.mpm_model.grid_dim_z = self.mpm_model.n_grid
        self.mpm_model.dx, self.mpm_model.inv_dx = self.mpm_model.grid_lim/self.mpm_model.n_grid, float(self.mpm_model.n_grid/self.mpm_model.grid_lim)

        self.mpm_model.E = 1000
        self.mpm_model.nu = 0.3
        self.mpm_model.mu = self.mpm_model.E / 2.0 / (1.0 + self.mpm_model.nu)
        self.mpm_model.lam = self.mpm_model.E * self.mpm_model.nu / (1.0 + self.mpm_model.nu) / (1.0 - 2.0 * self.mpm_model.nu)
        self.mpm_model.yield_stress = wp.zeros(shape=n_particles, dtype=float, device=device)
        self.mpm_model.material = 0

        self.mpm_model.friction_angle = 25.0
        sin_phi = wp.sin(self.mpm_model.friction_angle/180.0*3.14159265)
        self.mpm_model.alpha = wp.sqrt(2.0/3.0)*2.0*sin_phi/(3.0-sin_phi)
        self.mpm_model.gravitational_accelaration = 0.0

        self.mpm_state = MPMStateStruct()
        self.mpm_state.particle_x_initial = wp.empty(shape=n_particles, dtype=wp.vec3, device=device) # initial position
        self.mpm_state.particle_x = wp.empty(shape=n_particles, dtype=wp.vec3, device=device) # current position
        self.mpm_state.particle_u = wp.zeros(shape=n_particles, dtype=wp.vec3, device=device) # particle displacement
        self.mpm_state.particle_v = wp.zeros(shape=n_particles, dtype=wp.vec3, device=device) # particle velocity
        self.mpm_state.particle_F = wp.zeros(shape=n_particles, dtype=wp.mat33, device=device) # particle F elastic
        self.mpm_state.particle_F_trial = wp.zeros(shape=n_particles, dtype=wp.mat33, device=device) # apply return mapping will yield 
        self.mpm_state.particle_F_disp = wp.zeros(shape=n_particles, dtype=wp.mat33, device=device) # particle F elastic - ID
        self.mpm_state.particle_stress = wp.zeros(shape=n_particles, dtype=wp.mat33, device=device)
        self.mpm_state.particle_vol = wp.zeros(shape=n_particles, dtype=float, device=device) # particle volume
        self.mpm_state.particle_mass = wp.zeros(shape=n_particles, dtype=float, device=device) # particle mass
        self.mpm_state.particle_density = wp.zeros(shape=n_particles, dtype=float, device=device)
        self.mpm_state.particle_external_force = wp.zeros(shape=n_particles, dtype=wp.vec3, device=device)
        self.mpm_state.particle_C = wp.zeros(shape=n_particles, dtype=wp.mat33, device=device)
        self.mpm_state.particle_Jp = wp.zeros(shape=n_particles, dtype=float, device=device)

        self.mpm_state.grid_m = wp.zeros(shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid), dtype=float, device=device)
        self.mpm_state.grid_v_in = wp.zeros(shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid), dtype=wp.vec3, device=device)
        self.mpm_state.grid_v_out = wp.zeros(shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid), dtype=wp.vec3, device=device)

        self.particle_color = np.ones(shape=n_particles, dtype=float) # color for outputting obj, larger number -> whiter color
        self.time = 0.0

        self.grid_postprocess = []
        self.collider_params = []
        self.modify_bc = []

        self.tailored_struct_for_bc = MPMtailoredStruct()
        self.modify_bc_for_reduction = []

        self.mpm_state.neighboring_cells = wp.zeros(shape=(self.mpm_model.grid_dim_x, self.mpm_model.grid_dim_y, self.mpm_model.grid_dim_z), dtype=int, device=device)
        self.mpm_state.neighboring_cells_grid_v = wp.zeros(shape=(self.mpm_model.grid_dim_x, self.mpm_model.grid_dim_y, self.mpm_model.grid_dim_z), dtype=int, device=device)
        self.mpm_state.sample_particle_index_short = wp.zeros(shape=1, dtype=int, device=device)
        self.mpm_state.sample_particle_index_long = wp.zeros(shape=n_particles, dtype=int, device=device)
        self.mpm_state.sample_particle_all_index = wp.zeros(shape=n_particles, dtype=int, device=device)

        self.mpm_state.particle_color_temp = wp.zeros(shape=n_particles, dtype = int, device=device)

    def set_parameters(self,device = "cuda:0", **kwargs):
        if 'material' in kwargs:
            if kwargs['material'] == 'jelly': self.mpm_model.material = 0
            elif kwargs['material'] == 'metal': self.mpm_model.material = 1
            elif kwargs['material'] == 'sand': self.mpm_model.material = 2
            else: raise TypeError("Undefined material type")

        if 'grid_lim' in kwargs: self.mpm_model.grid_lim = kwargs['grid_lim']
        if 'n_grid' in kwargs: self.mpm_model.n_grid = kwargs['n_grid']
        self.mpm_model.grid_dim_x = self.mpm_model.n_grid
        self.mpm_model.grid_dim_y = self.mpm_model.n_grid
        self.mpm_model.grid_dim_z = self.mpm_model.n_grid
        self.mpm_model.dx, self.mpm_model.inv_dx = self.mpm_model.grid_lim/self.mpm_model.n_grid, float(self.mpm_model.n_grid/self.mpm_model.grid_lim)
        self.mpm_state.grid_m = wp.zeros(shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid), dtype=float, device=device)
        self.mpm_state.grid_v_in = wp.zeros(shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid), dtype=wp.vec3, device=device)
        self.mpm_state.grid_v_out = wp.zeros(shape=(self.mpm_model.n_grid, self.mpm_model.n_grid, self.mpm_model.n_grid), dtype=wp.vec3, device=device)

        if 'mu' in kwargs: self.mpm_model.mu = kwargs['mu']
        if 'lam' in kwargs: self.mpm_model.lam = kwargs['lam']
        if 'yield_stress' in kwargs: 
            val = kwargs['yield_stress']
            wp.launch(kernel = set_value_to_float_array, dim=self.n_particles,inputs=[self.mpm_model.yield_stress, val], device=device) 
        if 'hardening' in kwargs: self.mpm_model.hardening = kwargs['hardening']
        if 'xi' in kwargs: self.mpm_model.xi = kwargs['xi']
        if 'friction_angle' in kwargs: 
            self.mpm_model.friction_angle = kwargs['friction_angle']
            sin_phi = wp.sin(self.mpm_model.friction_angle/180.0*3.14159265)
            self.mpm_model.alpha = wp.sqrt(2.0/3.0)*2.0*sin_phi/(3.0-sin_phi)

        if 'g' in kwargs:
            self.mpm_model.gravitational_accelaration = kwargs['g']

        if 'density' in kwargs:
            density_value = kwargs['density']
            wp.launch(kernel = set_value_to_float_array, dim=self.n_particles,inputs=[self.mpm_state.particle_density, density_value], device=device)
            wp.launch(kernel = get_float_array_product, dim=self.n_particles, inputs=[self.mpm_state.particle_density, self.mpm_state.particle_vol, self.mpm_state.particle_mass], device=device)
        print("mean particle vol: ", torch.mean(wp.to_torch(self.mpm_state.particle_vol)))
        print("mean particle mass: ", torch.mean(wp.to_torch(self.mpm_state.particle_mass)))
        input()

    def load_from_sampling(self, sampling_h5, n_grid = 100, grid_lim = 1.0, device = "cuda:0"):
        if not os.path.exists(os.getcwd() + sampling_h5):
            print("h5 file cannot be found at ", os.getcwd() + sampling_h5)
            exit()

        h5file = h5py.File(os.getcwd() + sampling_h5, 'r')
        x, q, particle_mass, particle_volume, particle_density, f_tensor = h5file['x'], h5file['q'], h5file['masses'], h5file['particle_volume'], h5file['particle_density'], h5file['f_tensor']
        # f_tensor has shape (n, 3, 3)
        x, q = x[()].transpose(), q[()].transpose() # np vector of x # shape now is (n_particles, dim)
        self.dim, self.n_particles = q.shape[1], q.shape[0]
        self.initialize(self.n_particles, n_grid, grid_lim, device=device)
        print("Sampling particles are loaded from h5 file. Simulator is re-initialized for the correct n_particles")
        particle_mass = np.squeeze(particle_mass, 0)
        particle_density = np.squeeze(particle_density, 0)
        particle_volume = np.squeeze(particle_volume, 0)
        
        self.mpm_state.particle_x_initial = wp.from_numpy(x, dtype=wp.vec3, device=device) # initialize warp array from np
        self.mpm_state.particle_x = wp.from_numpy(x, dtype=wp.vec3, device=device) # initialize warp array from np
        wp.launch(kernel = set_vec3_to_zero, dim=self.n_particles,inputs=[self.mpm_state.particle_v], device=device) # set velocity to zero
        self.mpm_state.particle_F_disp = wp.from_numpy(f_tensor, dtype=wp.mat33, device=device)
        wp.launch(kernel = set_mat33_to_identity, dim=self.n_particles,inputs=[self.mpm_state.particle_F], device=device) # set elastic F to id
        self.mpm_state.particle_mass = wp.from_numpy(particle_mass, dtype=float, device=device)
        self.mpm_state.particle_density = wp.from_numpy(particle_density, dtype=float, device=device)
        self.mpm_state.particle_vol = wp.from_numpy(particle_volume, dtype=float, device=device)
        print("Particle position, F, mass, and density are loaded from initial sampling.")


    def p2g2p(self, step, dt, print_norm = False, device = "cuda:0"):
        wp.launch(kernel = update_x_from_u, dim=self.n_particles,inputs=[self.mpm_state], device=device)
        
        grid_size = (self.mpm_model.grid_dim_x, self.mpm_model.grid_dim_y, self.mpm_model.grid_dim_z)
        # print(grid_size)
        wp.launch(kernel = zero_grid, dim=(grid_size) ,inputs=[self.mpm_state, self.mpm_model], device=device)
        wp.launch(kernel = p2g_apic_stress_given, dim=self.n_particles,inputs=[self.mpm_state, self.mpm_model, dt], device=device) # apply p2g
        # print("grid v in: ", torch.norm(wp.to_torch(self.mpm_state.grid_v_in)))
        # print(wp.to_torch(self.mpm_state.grid_v_in).shape)
        # input()
        # print(torch.max(wp.to_torch(self.mpm_state.grid_v_in)))
        # print("grid mass: ", torch.max(wp.to_torch(self.mpm_state.grid_m)))
        # print(self.mpm_state.grid_m.shape)
        # print(self.mpm_state.grid_v_in.shape)
        # print(self.mpm_state.grid_v_out.shape)
        wp.launch(kernel = grid_normalization_and_gravity, dim=(grid_size) ,inputs=[self.mpm_state, self.mpm_model, dt], device=device)
        # print("grid v out: ", torch.norm(wp.to_torch(self.mpm_state.grid_v_out)))
        MA= torch.max(wp.to_torch(self.mpm_state.grid_v_out))
        print("max vout", MA)
        ID = torch.argmax(wp.to_torch(self.mpm_state.grid_v_out))
        # print(flatten2nD(ID, wp.to_torch(self.mpm_state.grid_v_out).shape))
        # temp = flatten2nD(ID, wp.to_torch(self.mpm_state.grid_v_out).shape)
        # print("point mass" , wp.to_torch(self.mpm_state.grid_m)[temp[0], temp[1], temp[2]])
        print("v_in at ID: ", wp.to_torch(self.mpm_state.grid_v_in).flatten()[ID])
        # print("mass at ID: ", wp.to_torch(self.mpm_state.grid_m).flatten()[ID])
        # print("v_out at ID: ", wp.to_torch(self.mpm_state.grid_v_out).flatten()[ID])
        for k in range(len(self.grid_postprocess)):
            wp.launch(kernel = self.grid_postprocess[k], dim=grid_size, inputs=[self.time, dt, self.mpm_state, self.mpm_model, self.collider_params[k]], device = device)
            if self.modify_bc[k] is not None:
                self.modify_bc[k](self.time, dt, self.collider_params[k])
        wp.launch(kernel = g2p_apic_and_stress, dim=self.n_particles,inputs=[self.mpm_state, self.mpm_model, dt], device=device)

        particle_v = wp.to_torch(self.mpm_state.particle_v)
        print("max particle v: ", torch.max(torch.abs(particle_v)).detach().cpu().numpy())
        print("max allowed  v: ", self.mpm_model.dx / dt)
        if torch.max(torch.abs(particle_v)).detach().cpu().numpy() > self.mpm_model.dx / dt:
            input()
        # print("C: ", torch.norm(wp.to_torch(self.mpm_state.particle_C)))
        # grid_v_in_tensor = wp.to_torch(self.mpm_state.grid_v_in)
        # print("norm of grid v in: ", "{:.4f}".format(float(torch.linalg.norm(grid_v_in_tensor))))
        wp.launch(kernel = update_u_from_x, dim=self.n_particles,inputs=[self.mpm_state], device=device)
        wp.launch(kernel = update_F_disp_from_F, dim=self.n_particles,inputs=[self.mpm_state], device=device)

        self.time = self.time + dt
        if print_norm:
            u_tensor, v_tensor, F_disp_tensor, stress = self.export_uvFstress_from_warp()
            print("at mpm step", step, end=" ")
            print("norm of u, v, F_disp, stress, respectively: ", "{:.4f}".format(float(torch.linalg.norm(u_tensor))), "{:.4f}".format(float(torch.linalg.norm(v_tensor))), "{:.4f}".format(float(torch.linalg.norm(F_disp_tensor))), "{:.4f}".format(float(torch.linalg.norm(stress))))

    

    def experiment_for_C(self, device = "cuda:0"):
        wp.launch(kernel = update_neighboring_cells_for_grid_v, dim=self.mpm_state.sample_particle_index_short.shape[0],inputs=[self.mpm_state, self.mpm_model], device=device)
        return wp.to_torch(self.mpm_state.neighboring_cells_grid_v)



    # reduced p2g2p for Lagrangian view, p2g is LOOPED over all particles and then with an IF
    def p2g2p_reduced(self, step, dt, device = "cuda:0"):
        grid_size = (self.mpm_model.grid_dim_x, self.mpm_model.grid_dim_y, self.mpm_model.grid_dim_z)
        wp.launch(kernel = update_x_from_u, dim=self.n_particles,inputs=[self.mpm_state], device=device)
        wp.launch(kernel = zero_grid, dim=(grid_size) ,inputs=[self.mpm_state, self.mpm_model], device=device)
        wp.launch(kernel = update_neighboring_cells, dim=self.mpm_state.sample_particle_index_short.shape[0],inputs=[self.mpm_state, self.mpm_model], device=device)
        wp.launch(kernel = p2g_sample_all, dim=self.n_particles,inputs=[self.mpm_state, self.mpm_model, dt], device=device) # apply p2g
        #------------------------------ a generalized BC handling that will always work ---------------------
        wp.launch(kernel = grid_normalization_and_gravity_sample_all, dim=(grid_size) ,inputs=[self.mpm_state, self.mpm_model, dt], device=device)  
        for k in range(len(self.grid_postprocess)):
            wp.launch(kernel = self.grid_postprocess[k], dim=grid_size, inputs=[self.time, dt, self.mpm_state, self.mpm_model, self.collider_params[k]], device = device)
            if self.modify_bc[k] is not None:
                self.modify_bc[k](self.time, dt, self.collider_params[k])
        #------------------------------ a generalized BC handling that will always work ---------------------
        self.time = self.time + dt

    def load_x_initial_from_tensor(self, tensor_x_iniital, device = "cuda:0"):
        self.mpm_state.particle_x_initial = torch2warp_vec3(tensor_x_iniital, dvc=device)
    def load_mass_from_tensor(self, tensor_mass, device = "cuda:0"):
        self.mpm_state.particle_mass = wp.from_torch(tensor_mass, dvc = device)
    
    def load_uvF_from_tensor(self, tensor_u, tensor_v, tensor_F, clone = True, device = "cuda:0"):
        # clone should be needed if using nonlinear projection -- subject to change
        if clone:
            tensor_u = tensor_u.clone().detach()
            tensor_v = tensor_v.clone().detach()
            tensor_F = tensor_F.clone().detach()
        # input_tensor is (n_particles, dim)
        # tensor_F is of shape (n, 9)
        tensor_F = torch.reshape(tensor_F, (-1,3,3)) # arranged by rowmajor
        self.mpm_state.particle_u = torch2warp_vec3(tensor_u, dvc = device)
        self.mpm_state.particle_v = torch2warp_vec3(tensor_v, dvc = device)
        self.mpm_state.particle_F_disp = torch2warp_mat33(tensor_F, dvc = device)

    def load_C_from_tensor(self, tensor_C, clone = True, device = "cuda:0"):
        if clone:
            tensor_C = tensor_C.clone().detach()
        tensor_C = torch.reshape(tensor_C, (-1,3,3)) # arranged by rowmajor
        self.mpm_state.particle_C = torch2warp_mat33(tensor_C, dvc=device)

    def load_uvstress_from_tensor(self, tensor_u, tensor_v, tensor_stress, clone = True, device = "cuda:0"):
        # clone should be needed if using nonlinear projection -- subject to change
        if clone:
            tensor_u = tensor_u.clone().detach()
            tensor_v = tensor_v.clone().detach()
            tensor_stress = tensor_stress.clone().detach()
        # input_tensor is (n_particles, dim)
        # tensor_F is of shape (n, 9)
        tensor_stress = torch.reshape(tensor_stress, (-1,3,3)) # arranged by rowmajor
        self.mpm_state.particle_u = torch2warp_vec3(tensor_u, dvc = device)
        self.mpm_state.particle_v = torch2warp_vec3(tensor_v, dvc = device)
        # print("shape of stress tensor: ", tensor_stress.shape)
        # input()
        self.mpm_state.particle_stress = torch2warp_mat33(tensor_stress, dvc = device)

    def load_uvFstress_from_h5(self, h5filename, sample_point = 0, device = "cuda:0"):
        if not os.path.exists(os.getcwd() + h5filename):
            print("h5 file cannot be found at ", os.getcwd() + h5filename)
            exit()

        h5file = h5py.File(os.getcwd() + h5filename, 'r')
        q = h5file['q']
        v = h5file['v']
        f = h5file['f_tensor']
        stress = h5file['stress']
        q = q[()].transpose()
        v = v[()].transpose()
        f = f[()].transpose()
        stress = stress[()].transpose() # n, 6 # .transpose()
        # print(stress.shape)
        # input()
        f = np.reshape(f, (f.shape[0], 3, 3))
        stress_in = np.zeros((f.shape[0], 3, 3))
        stress_in[:, 0,0] = stress[:, 0]
        stress_in[:, 0,1] = stress[:, 1]
        stress_in[:, 0,2] = stress[:, 2]
        stress_in[:, 1,0] = stress[:, 1]
        stress_in[:, 1,1] = stress[:, 3]
        stress_in[:, 1,2] = stress[:, 4]
        stress_in[:, 2,0] = stress[:, 2]
        stress_in[:, 2,1] = stress[:, 4]
        stress_in[:, 2,2] = stress[:, 5]
        
        # print(v[0, 0])
        # print(float("{:.4f}".format(v[sample_point, 1]))) # second entry of v
        # print(float("{:.4f}".format(f[sample_point, 0,0])))
        self.mpm_state.particle_u = wp.from_numpy(q, dtype=wp.vec3, device=device)
        self.mpm_state.particle_v = wp.from_numpy(v, dtype=wp.vec3, device=device)
        # self.mpm_state.particle_F_disp = wp.from_numpy(f, dtype=wp.mat33, device=device)
        self.mpm_state.particle_stress = wp.from_numpy(stress_in, dtype=wp.mat33, device=device)
        print("stress in:", np.linalg.norm((stress_in)))

    def export_uvFstress_from_warp(self, device = "cuda:0"):
        u_tensor = wp.to_torch(self.mpm_state.particle_u)
        v_tensor = wp.to_torch(self.mpm_state.particle_v)
        F_tensor = wp.to_torch(self.mpm_state.particle_F_disp)
        stress = wp.to_torch(self.mpm_state.particle_stress)
        return u_tensor, v_tensor, F_tensor, stress

    def particle_position2obj(self, fullfilename):
        if os.path.exists(fullfilename):
            os.remove(fullfilename)
        objfile = open(fullfilename, 'w')
        current_position = self.mpm_state.particle_x.numpy()
        for i in range(self.n_particles):
            color = 1.0 - 1.0/self.particle_color[i]
            line =  "v " + str(current_position[i][0]) + " " + str(current_position[i][1]) + " " + str(current_position[i][2]) + " " + str(color) + " " + str(color) + " " + str(color)
            objfile.write(line)
            objfile.write('\n')
        print('warp mpm solver writes current position at ', fullfilename)

    def particle_F_norm2obj(self, fullfilename, F_upperbound = 2.0):
        if os.path.exists(fullfilename):
            os.remove(fullfilename)
        objfile = open(fullfilename, 'w')
        original_position = self.mpm_state.particle_x_initial.numpy()
        F_disp_np = self.mpm_state.particle_F_disp.numpy().reshape(-1,9)
        F_disp_norm = np.linalg.norm(F_disp_np, axis=1)
        
        for i in range(self.n_particles):
            color = np.minimum(1.0,F_disp_norm[i]/F_upperbound)
            line =  "v " + str(original_position[i][0]) + " " + str(original_position[i][1]) + " " + str(original_position[i][2]) + " " + str(color) + " 0 0" 
            objfile.write(line)
            objfile.write('\n')


    def save_data_at_frame(self, dir_name, frame, include_parameter = 'friction_angle', save_to_h5 = True, save_to_obj = True, save_visual_F= False):
        os.umask(0)
        os.makedirs(dir_name, 0o777, exist_ok=True)
        fullfilename = dir_name + '/h5_f_' + str(frame).zfill(10) + '.h5'
        fullfilename_obj = fullfilename[:-2]+'obj'
        fullfilename_F = dir_name + '/visualF_f_' + str(frame).zfill(10) + '.obj'
        if save_to_obj:
            if os.path.exists(fullfilename_obj): os.remove(fullfilename_obj)
            self.particle_position2obj(fullfilename_obj)
        if save_visual_F:
            self.particle_F_norm2obj(fullfilename_F)
        #----------- Initial position --------------
        x_initial_np = self.mpm_state.particle_x_initial.numpy()
        x_initial_np = x_initial_np.transpose()
        #----------- Displacement --------------
        u_np = self.mpm_state.particle_u.numpy() # x_np has dimension (dim, n_particles)
        u_np = u_np.transpose()
        #----------- velocity --------------
        v_np = self.mpm_state.particle_v.numpy() # x_np has dimension (dim, n_particles)
        v_np = v_np.transpose()
        #----------- Time --------------
        currentTime = np.array([self.time])
        currentTime = currentTime.reshape(1,1) # need a 1by1 matrix
        #----------- Particle mass --------------
        p_mass_np = self.mpm_state.particle_mass.numpy()
        p_mass_np = p_mass_np.reshape(1, self.n_particles)
        #----------- Particle volume ------------
        # p_volume_np = self.particle_volume.numpy()
        # p_volume_np = p_volume_np.reshape(1, self.n_particles)
        
        #----------- Deformation gradient, dudX --------------
        f_tensor_np = self.mpm_state.particle_F_disp.numpy() # dimension (n_particles, 3, 3)
        f_tensor_np = f_tensor_np.reshape(-1,9) # (n,3,3) -> (n,9), row_major
        f_tensor_np = f_tensor_np.transpose() # (9,n)

        #----------- Particle C --------------
        C_np = self.mpm_state.particle_C.numpy() # dimension (n_particles, 3, 3)
        C_np = C_np.reshape(-1,9) # (n,3,3) -> (n,9), row_major
        C_np = C_np.transpose() # (9,n)

        #----------- stress = stress(F) ------- 
        stress_np = self.mpm_state.particle_stress.numpy() # [0 1 2; 3 4 5; 6 7 8]
        stress_np = stress_np.reshape(-1,9) # (n,3,3) -> (n,9), row_major
        print("stress out:", np.linalg.norm((stress_np)))
        stress_np = stress_np[:, [0,1,2,4,5,8]]
        stress_np = stress_np.transpose()

        if include_parameter == 'friction_angle':
            mu = np.ones((1, self.n_particles)) * self.mpm_model.friction_angle#np.ndarray(self.mpm_model.friction_angle)
        
        # print(stress_np.shape)
        # input()
        if save_to_h5:
            if os.path.exists(fullfilename): os.remove(fullfilename)
            newFile = h5py.File(fullfilename, "w")
            newFile.create_dataset("x", data=x_initial_np) # initial position
            newFile.create_dataset("q", data=u_np) # displacement
            newFile.create_dataset("time", data=currentTime) # current time
            newFile.create_dataset("masses", data=p_mass_np) # particle mass
            newFile.create_dataset("f_tensor", data=f_tensor_np) # deformation grad
            newFile.create_dataset("stress", data=stress_np) # deformation grad
            newFile.create_dataset("v", data=v_np) # particle velocity
            if include_parameter:
                newFile.create_dataset("mu", data=mu) # particle velocity
            newFile.create_dataset("C", data=C_np) # particle velocity
            print("save siumlation data at frame ", frame, " to ", fullfilename)


    def add_surface_collider(self, point, normal, surface="sticky", friction=0.0, start_time=0.0, end_time=999.0):

        point = list(point)
        # Normalize normal
        normal_scale = 1.0 / wp.sqrt(sum(x**2 for x in normal))
        normal = list(normal_scale * x for x in normal)

        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time

        collider_param.point = wp.vec3(point[0], point[1], point[2])
        collider_param.normal = wp.vec3(normal[0], normal[1], normal[2])
        
        if surface == "sticky" and friction != 0: raise ValueError('friction must be 0 on sticky surfaces.')
        if surface == "sticky": collider_param.surface_type = 0
        elif surface == "slip": collider_param.surface_type = 1
        else: collider_param.surface_type = 2
            # frictional
        collider_param.friction = friction

        self.collider_params.append(collider_param)
        @wp.kernel
        def collide(time: float, dt: float, state: MPMStateStruct, model: MPMModelStruct, param: Dirichlet_collider):
            grid_x, grid_y, grid_z = wp.tid()
            if time>=param.start_time and time<param.end_time:
                offset = wp.vec3(float(grid_x)*model.dx-param.point[0], float(grid_y)*model.dx-param.point[1], float(grid_z)*model.dx-param.point[2])
                n = wp.vec3(param.normal[0], param.normal[1], param.normal[2])
                dotproduct = wp.dot(offset, n)

                if dotproduct < 0.0:
                    if param.surface_type == 0:
                        state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0,0.0,0.0)
                    else:
                        v = state.grid_v_out[grid_x, grid_y, grid_z]
                        normal_component = wp.dot(v, n)
                        if param.surface_type == 1:
                            v = v - normal_component * n # Project out all normal component
                        else: 
                            v = v - wp.min(normal_component, 0.0) * n # Project out only inward normal component
                        if normal_component<0.0 and wp.length(v) > 1e-20:
                            v = wp.max(0.0, wp.length(v) + normal_component * param.friction) * wp.normalize(v) # apply friction here
                        state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0,0.0,0.0)
        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)

    def add_surface_collider2(self, point, normal, surface="sticky", friction=0.0, start_time=0.0, end_time=999.0):

        point = list(point)
        # Normalize normal
        normal_scale = 1.0 / wp.sqrt(sum(x**2 for x in normal))
        normal = list(normal_scale * x for x in normal)

        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time

        collider_param.point = wp.vec3(point[0], point[1], point[2])
        collider_param.normal = wp.vec3(normal[0], normal[1], normal[2])
        
        if surface == "sticky" and friction != 0: raise ValueError('friction must be 0 on sticky surfaces.')
        if surface == "sticky": collider_param.surface_type = 0
        elif surface == "slip": collider_param.surface_type = 1
        else: collider_param.surface_type = 2
            # frictional
        collider_param.friction = friction

        self.collider_params.append(collider_param)
        @wp.kernel
        def collide(time: float, dt: float, state: MPMStateStruct, model: MPMModelStruct, param: Dirichlet_collider):
            grid_x, grid_y, grid_z = wp.tid()
            if time>=param.start_time and time<param.end_time:
                offset = wp.vec3(float(grid_x)*model.dx-param.point[0], float(grid_y)*model.dx-param.point[1], float(grid_z)*model.dx-param.point[2])
                n = wp.vec3(param.normal[0], param.normal[1], param.normal[2])
                dotproduct = wp.dot(offset, n)

                if dotproduct < 0.0:
                    if param.surface_type == 0:
                        state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0,0.0,0.0)
                    else:
                        v = state.grid_v_out[grid_x, grid_y, grid_z]
                        normal_component = wp.dot(v, n)
                        if param.surface_type == 1:
                            v = v - normal_component * n # Project out all normal component
                        else: 
                            v = v - wp.min(normal_component, 0.0) * n # Project out only inward normal component
                        if normal_component<0.0 and wp.length(v) > 1e-20:
                            v = wp.max(0.0, wp.length(v) + normal_component * param.friction) * wp.normalize(v) # apply friction here
                        state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0,0.0,0.0)
        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)

    def add_bounding_box(self, start_time = 0.0, end_time = 999.0):
        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time

        self.collider_params.append(collider_param)
        @wp.kernel
        def collide(time: float, dt: float, state: MPMStateStruct, model: MPMModelStruct, param: Dirichlet_collider):
            grid_x, grid_y, grid_z = wp.tid()
            padding = 3
            if time>=param.start_time and time<param.end_time:
                if grid_x < padding and state.grid_v_out[grid_x, grid_y, grid_z][0] < 0:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0, state.grid_v_out[grid_x, grid_y, grid_z][1], state.grid_v_out[grid_x, grid_y, grid_z][2])
                if grid_x >= model.grid_dim_x - padding and state.grid_v_out[grid_x, grid_y, grid_z][0] > 0:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0, state.grid_v_out[grid_x, grid_y, grid_z][1], state.grid_v_out[grid_x, grid_y, grid_z][2])

                if grid_y < padding and state.grid_v_out[grid_x, grid_y, grid_z][1] < 0:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(state.grid_v_out[grid_x, grid_y, grid_z][0], 0.0, state.grid_v_out[grid_x, grid_y, grid_z][2])
                if grid_y >= model.grid_dim_y - padding and state.grid_v_out[grid_x, grid_y, grid_z][1] > 0:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(state.grid_v_out[grid_x, grid_y, grid_z][0], 0.0, state.grid_v_out[grid_x, grid_y, grid_z][2])

                if grid_z < padding and state.grid_v_out[grid_x, grid_y, grid_z][2] < 0:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(state.grid_v_out[grid_x, grid_y, grid_z][0], state.grid_v_out[grid_x, grid_y, grid_z][1], 0.0)
                if grid_z >= model.grid_dim_z - padding and state.grid_v_out[grid_x, grid_y, grid_z][2] > 0:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(state.grid_v_out[grid_x, grid_y, grid_z][0], state.grid_v_out[grid_x, grid_y, grid_z][1], 0.0)
        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)

    def set_velocity_on_infinite_plane(self, point, normal, velocity, threshold, start_time=0.0, end_time=999.0, reset = 0):
        point = list(point)
        # Normalize normal
        normal_scale = 1.0 / wp.sqrt(sum(x**2 for x in normal))
        normal = list(normal_scale * x for x in normal)

        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time
        collider_param.point = wp.vec3(point[0], point[1], point[2])
        collider_param.normal = wp.vec3(normal[0], normal[1], normal[2])
        collider_param.velocity = wp.vec3(velocity[0], velocity[1], velocity[2])
        collider_param.threshold = threshold
        collider_param.reset = reset
        self.collider_params.append(collider_param)
        
        @wp.kernel
        def collide(time: float, dt: float, state: MPMStateStruct, model: MPMModelStruct, param: Dirichlet_collider):
            grid_x, grid_y, grid_z = wp.tid()
            if time>=param.start_time and time<param.end_time:
                offset = wp.vec3(float(grid_x)*model.dx-param.point[0], float(grid_y)*model.dx-param.point[1], float(grid_z)*model.dx-param.point[2])
                n = wp.vec3(param.normal[0], param.normal[1], param.normal[2])
                dotproduct = wp.dot(offset, n)
                if wp.abs(dotproduct) <= param.threshold:
                    state.grid_v_out[grid_x, grid_y, grid_z] = param.velocity
            elif param.reset == 1:
                if time<param.end_time + 1.5 * dt * 10.0:
                    offset = wp.vec3(float(grid_x)*model.dx-param.point[0], float(grid_y)*model.dx-param.point[1], float(grid_z)*model.dx-param.point[2])
                    n = wp.vec3(param.normal[0], param.normal[1], param.normal[2])
                    dotproduct = wp.dot(offset, n)
                    if wp.abs(dotproduct) <= param.threshold:
                        state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0,0.0,0.0)

        def modify(time, dt, param: Dirichlet_collider):
            
            if time>=param.start_time and time<param.end_time:
                param.point = wp.vec3(param.point[0]+dt*param.velocity[0], param.point[1]+dt*param.velocity[1], param.point[2]+dt*param.velocity[2])# param.point + dt * param.velocity
                # print(param.point[0])
        self.grid_postprocess.append(collide)
        self.modify_bc.append(modify)

    def set_column_moving_as_circle(self, R, radius, height, v_scale, start_time = 0.0, end_time = 999.0):
        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time
        collider_param.point = wp.vec3(0.5, 0.069, 0.5+R) # center of bottom
        collider_param.R = R
        collider_param.radius = radius
        collider_param.height = height
        collider_param.v_scale = v_scale
        self.collider_params.append(collider_param)
        @wp.kernel
        def collide(time: float, dt: float, state: MPMStateStruct, model: MPMModelStruct, param: Dirichlet_collider):
            grid_x, grid_y, grid_z = wp.tid()
            if time>=param.start_time and time<param.end_time:
                offset = wp.vec3(float(grid_x)*model.dx-param.point[0], float(grid_y)*model.dx-param.point[1], float(grid_z)*model.dx-param.point[2])
                
                if offset[0]*offset[0]+offset[2]*offset[2] < param.radius*param.radius:
                    if wp.abs(offset[1])<param.height:
                        Z = wp.vec3(param.point[0]-0.5, param.point[1], param.point[2]-0.5)
                        dp = wp.dot(Z, wp.vec3(1.0,0.0,0.0)) / wp.length(Z)
                        rdoty = wp.dot(Z, wp.vec3(0.0,0.0,1.0))
                        theta = wp.acos(dp)
                        if rdoty < 0:
                            theta = -theta
                        state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(-param.v_scale * wp.sin(theta),0.0, param.v_scale * wp.cos(theta))

        def modify(time, dt, param: Dirichlet_collider):
            if time>=param.start_time and time<param.end_time:
                Z = wp.vec3(param.point[0]-0.5, param.point[1], param.point[2]-0.5)
                dp = wp.dot(Z, wp.vec3(1.0,0.0,0.0)) / wp.length(Z)
                rdoty = wp.dot(Z, wp.vec3(0.0,0.0,1.0))
                theta = wp.acos(dp)
                if rdoty < 0:
                    theta = -theta
                V = wp.vec3(-v_scale * wp.sin(theta),0.0, v_scale * wp.cos(theta))
                param.point = wp.vec3(param.point[0]+dt*V[0], param.point[1]+dt*V[1], param.point[2]+dt*V[2])# param.point + dt * param.velocity
        self.grid_postprocess.append(collide)
        self.modify_bc.append(modify)

    def set_rotation_on_disk(self, point, normal, radius, width, v_scale, start_time=0.0, end_time=999.0, reset = 0):
        point = list(point)
        normal_scale = 1.0 / wp.sqrt(sum(x**2 for x in normal))
        normal = list(normal_scale * x for x in normal)
        k = 0
        while k < 3:
            if abs(normal[k]) > 1e-1:
                break
        local_axis_x = [1,1,1]
        local_axis_x[k] = - (normal[0] +normal[1]+ normal[2] - normal[k]) / normal[k]
        local_axis_y = list(np.cross(np.array(normal), np.array(local_axis_x)))
        normal_scale = 1.0 / wp.sqrt(sum(x**2 for x in local_axis_x))
        local_axis_x = list(normal_scale * x for x in local_axis_x)
        normal_scale = 1.0 / wp.sqrt(sum(x**2 for x in local_axis_y))
        local_axis_y = list(normal_scale * x for x in local_axis_y)
        collider_param = Dirichlet_collider()
        collider_param.point = wp.vec3(point[0], point[1], point[2])
        collider_param.normal = wp.vec3(normal[0], normal[1], normal[2])
        collider_param.x_unit = wp.vec3(local_axis_x[0], local_axis_x[1], local_axis_x[2])
        collider_param.y_unit = wp.vec3(local_axis_y[0], local_axis_y[1], local_axis_y[2])
        collider_param.radius = radius
        collider_param.v_scale = v_scale
        collider_param.width = width
        collider_param.start_time = start_time
        collider_param.end_time = end_time
        collider_param.reset = reset

        self.collider_params.append(collider_param)
        @wp.kernel
        def collide(time: float, dt: float, state: MPMStateStruct, model: MPMModelStruct, param: Dirichlet_collider):
            grid_x, grid_y, grid_z = wp.tid()
            offset = wp.vec3(float(grid_x)*model.dx-param.point[0], float(grid_y)*model.dx-param.point[1], float(grid_z)*model.dx-param.point[2])
            n = wp.vec3(param.normal[0], param.normal[1], param.normal[2])
            dotproduct = wp.dot(offset, n) 
            if time>=param.start_time and time<param.end_time:
                if wp.abs(dotproduct) < param.width:
                    # print(offset)
                    # r = wp.vec3(offset[0] - dotproduct*n[0], offset[1] - dotproduct*n[1], offset[2] - dotproduct*n[2])
                    r = wp.vec3(0.0, offset[1], offset[2])
                    if wp.length(r) < param.radius:
                        dp = wp.dot(r, param.x_unit) / wp.length(r)
                        rdoty = wp.dot(r, param.y_unit)
                        theta = wp.acos(dp)
                        if rdoty < 0:
                            theta = -theta
                        # print(theta)
                        # state.grid_v_out[grid_x, grid_y, grid_z]
                        # sc = (0.5+(0.5*(param.width-wp.abs(dotproduct))/param.width))
                        sca_x = - wp.length(r) * wp.sin(theta) * param.v_scale * 1.0
                        sca_y = wp.length(r) * wp.cos(theta) * param.v_scale * 1.0
                        state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(sca_x*param.x_unit[0]+sca_y*param.y_unit[0], sca_x*param.x_unit[1]+sca_y*param.y_unit[1], sca_x*param.x_unit[2]+sca_y*param.y_unit[2])
            elif param.reset == 1:
                if time<param.end_time + 1.5 * dt:
                
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0,0.0,0.0)

        self.grid_postprocess.append(collide)
        self.modify_bc.append(None)

    def set_velocity_on_cube(self, point, width, velocity, start_time=0.0, end_time=999.0, reset = 0):
        point = list(point)

        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time
        collider_param.point = wp.vec3(point[0], point[1], point[2])
        collider_param.width = width
        collider_param.velocity = wp.vec3(velocity[0], velocity[1], velocity[2])
        # collider_param.threshold = threshold
        collider_param.reset = reset
        self.collider_params.append(collider_param)
        @wp.kernel
        def collide(time: float, dt: float, state: MPMStateStruct, model: MPMModelStruct, param: Dirichlet_collider):
            grid_x, grid_y, grid_z = wp.tid()
            if time>=param.start_time and time<param.end_time:
                offset = wp.vec3(float(grid_x)*model.dx-param.point[0], float(grid_y)*model.dx-param.point[1], float(grid_z)*model.dx-param.point[2])
                if wp.abs(offset[0]) < param.width and wp.abs(offset[1]) < param.width and wp.abs(offset[2]) < param.width:
                    state.grid_v_out[grid_x, grid_y, grid_z] = param.velocity
            elif param.reset == 1:
                if time<param.end_time + 1.5 * dt:
                    state.grid_v_out[grid_x, grid_y, grid_z] = wp.vec3(0.0,0.0,0.0)
        def modify(time, dt, param: Dirichlet_collider):
            
            if time>=param.start_time and time<param.end_time:
                param.point = wp.vec3(param.point[0]+dt*param.velocity[0], param.point[1]+dt*param.velocity[1], param.point[2]+dt*param.velocity[2])# param.point + dt * param.velocity
        self.grid_postprocess.append(collide)
        self.modify_bc.append(modify)

    def add_sphere_leader(self, center, radius, velocity, start_time = 0, end_time = 999):
        center = list(center)
        velocity = list(velocity)
        collider_param = Dirichlet_collider()
        collider_param.start_time = start_time
        collider_param.end_time = end_time
        collider_param.point = wp.vec3(center[0], center[1], center[2])
        collider_param.velocity = wp.vec3(velocity[0], velocity[1], velocity[2])
        collider_param.radius = radius

        self.collider_params.append(collider_param)

        @wp.kernel
        def collide(time: float, dt: float, state: MPMStateStruct, model: MPMModelStruct, param: Dirichlet_collider):
            grid_x, grid_y, grid_z = wp.tid()
            if time>=param.start_time and time<param.end_time:
                offset = wp.vec3(float(grid_x)*model.dx-param.point[0], float(grid_y)*model.dx-param.point[1], float(grid_z)*model.dx-param.point[2])
                if wp.length(offset) < param.radius:
                    state.grid_v_out[grid_x, grid_y, grid_z] = param.velocity
                    
        def modify(time, dt, param: Dirichlet_collider):
            if time>=param.start_time and time<param.end_time:
                param.point = wp.vec3(param.point[0]+dt*param.velocity[0], param.point[1]+dt*param.velocity[1], param.point[2]+dt*param.velocity[2])# param.point + dt * param.velocity
        
        self.grid_postprocess.append(collide)
        self.modify_bc.append(modify)