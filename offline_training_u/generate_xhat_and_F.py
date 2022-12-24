from multiprocessing import Pool
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support
import os
import h5py
import numpy as np
import os
import re
from numpy import linalg as LA

# fetch existing label from reconstruction

def extract_h5_files(path):
    # return a list of filenames, they are all h5 files under path
    # [path/subfolder/xxx.h5, ...]
    config_file_pattern = r'h5_f_(\d+)\.h5'
    config_file_matcher = re.compile(config_file_pattern)
    dir_pattern = r'sim_seq_(.*?)'
    dir_matcher = re.compile(dir_pattern)
    data_list = []
    dir_list = os.listdir(path)
    num_sims = 0
    dir_list_sim = []
    for dirname in dir_list:
        if os.path.isdir(os.path.join(path,dirname)):
            dir_match = dir_matcher.match(
                dirname)
            if dir_match != None:
                num_sims += 1
                dir_list_sim.append(dirname)

    for dirname in dir_list_sim:
        for filename in os.listdir(os.path.join(path, dirname)):
            config_file_match = config_file_matcher.match(
                filename)
            if config_file_match is None:
                continue
            fullfilename = os.path.join(path, dirname, filename)
            data_list.append(fullfilename)
    return data_list


def secondary_training_data_onefile(ground_truth_file_name, reconstruction_folder_name, new_target_name):
    ground_truth_file_name = os.path.normpath(ground_truth_file_name)
    ground_truth_file = h5py.File(ground_truth_file_name, 'r')
    
    reconstruct_path_list = ground_truth_file_name.split(os.sep)
    reconstruct_path_list[-3] = reconstruction_folder_name

    reconstruction_file_name = os.path.join(*reconstruct_path_list)
    reconstruction_file = h5py.File(reconstruction_file_name, 'r')

    newfile_path_list = ground_truth_file_name.split(os.sep)
    newfile_path_list[-3] = newfile_path_list[-3] + "_xlabel_at_" + reconstruction_folder_name[-15:]+ "_and_" + new_target_name
    
    dir_name = newfile_path_list[:-1]
    dir_name = os.path.join(*dir_name)
    os.makedirs(dir_name, exist_ok=True)
    newfilename = os.path.join(*newfile_path_list)

    if os.path.exists(newfilename):
        os.remove(newfilename)
    newFile = h5py.File(newfilename, "w")

    tempX = ground_truth_file["x"]
    tempX = tempX[()]
    newFile.create_dataset("x", data=tempX) 
    
    tempT = ground_truth_file["time"]
    tempT = tempT[()]
    newFile.create_dataset("time", data=tempT)

    tempQ = ground_truth_file[new_target_name]
    tempQ = tempQ[()]
    newFile.create_dataset("q", data=tempQ)
    

    tempLabel = reconstruction_file["label"]
    tempLabel = tempLabel[()]
    tempLabel = np.repeat(tempLabel, tempQ.shape[1], 1) # shape = (lbl, n_particle)
    newFile.create_dataset("prev_label", data=tempLabel)

    print("Finish writing label and ", new_target_name, "to ", newfilename)


if __name__ == '__main__':
    pool = Pool(os.cpu_count())
    print("Pred data in mpm-data/training_data will be fetched")
    input("Check testcase name !!!")
    input("Check reconstruction_folder name!!")
    input("Check new target name! it will be saved as q in the newfile")
    parent_dir = "../mpm-data/training_data/"
    
    testcase_name = "vm_unit_apic"
    reconstruction_folder_name = "pred_vm_unit_apic_train_q_at_20221204-150445"
    new_target_name = 'f_tensor'

    
    ground_truth_data_list = extract_h5_files(parent_dir + testcase_name)
    
    # secondary_training_data_onefile(ground_truth_data_list[0], reconstruction_folder_name, new_target_name)
    pool.starmap(secondary_training_data_onefile, zip(ground_truth_data_list, repeat(reconstruction_folder_name), repeat(new_target_name)))


