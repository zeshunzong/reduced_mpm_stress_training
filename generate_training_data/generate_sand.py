import warp as wp
import torch
import numpy as np
import h5py
import time
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir) 
from engine_stress.mpm_solver_warp import MPM_Simulator_WARP


wp.init()
wp.config.verify_cuda = True
mpm_solver = MPM_Simulator_WARP(10)
mpm_solver.load_from_sampling("/initial_data/sand_column_large/sand_column_large170_initial_sampling.h5", n_grid = 200)
print(mpm_solver.n_particles)

input()
angle = 30
mpm_solver.set_parameters(mu = 100000.0, lam = 0.3, material='sand', friction_angle = angle, g = 9.8, density = 2200.0)
mpm_solver.add_surface_collider((0.0, 0.58, 0.0), (0.0,1.0,0.0), 'sticky', 0.0)

mpm_solver.add_bounding_box()
wp.synchronize()
mpm_solver.save_data_at_frame("../mpm-data/training_data/sand_stress_training/sim_seq_" + str(angle).zfill(2), 0)
for k in range(1,400):
    mpm_solver.p2g2p(k, 0.0008, print_norm=True)
    # input()
    mpm_solver.save_data_at_frame("../mpm-data/training_data/sand_stress_training/sim_seq_" + str(angle).zfill(2), k, save_to_obj=True, save_to_h5=True, save_visual_F=False)


   
   