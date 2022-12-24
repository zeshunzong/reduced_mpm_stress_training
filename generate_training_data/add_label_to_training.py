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


def secondary_training_data_onefile(ground_truth_file_name, reconstruction_folder_name):
    ground_truth_file_name = os.path.normpath(ground_truth_file_name)
    ground_truth_file = h5py.File(ground_truth_file_name, 'r')
    
    reconstruct_path_list = ground_truth_file_name.split(os.sep)
    reconstruct_path_list[-3] = reconstruction_folder_name

    reconstruction_file_name = os.path.join(*reconstruct_path_list)
    # reconstruction_file = h5py.File(reconstruction_file_name, 'r')
    if os.path.exists(reconstruction_file_name):
        reconstruction_file = h5py.File(reconstruction_file_name, 'r')

        newfile_path_list = ground_truth_file_name.split(os.sep)
        newfile_path_list[-3] = newfile_path_list[-3] + "_and_xlabel_at_" + reconstruction_folder_name[-15:]# + "_and_" + new_target_name
        
        dir_name = newfile_path_list[:-1]
        dir_name = os.path.join(*dir_name)
        os.makedirs(dir_name, exist_ok=True)
        newfilename = os.path.join(*newfile_path_list)

        if os.path.exists(newfilename):
            os.remove(newfilename)
        newFile = h5py.File(newfilename, "w")


        list_of_datasets_names = list(ground_truth_file.keys())
        for k in list_of_datasets_names:
            original_ds = ground_truth_file[k]
            original_ds = original_ds[()]
            newFile.create_dataset(k, data=original_ds) 

        tempX = ground_truth_file["x"]
        tempX = tempX[()]
        

        tempLabel = reconstruction_file["label"]
        tempLabel = tempLabel[()]
        tempLabel = np.repeat(tempLabel, tempX.shape[1], 1) # shape = (lbl, n_particle)
        newFile.create_dataset("prev_label", data=tempLabel)

        print("Finish writing label and to ", newfilename)
    else:
        print("meet a file from training data that is not trained/reconstructed")


if __name__ == '__main__':
    pool = Pool(os.cpu_count())
    print("Pred data in mpm-data/training_data will be fetched")
    
    parent_dir = "../mpm-data/training_data/"
    
    testcase_name = "sand_debug_large_stress_mu2"
    reconstruction_folder_name = "pred_sand_debug_large_stress_mu2_train_q_at_20221219-101027"
    # new_target_name = 'stress'

    
    ground_truth_data_list = extract_h5_files(parent_dir + testcase_name)
    
    # secondary_training_data_onefile(ground_truth_data_list[0], reconstruction_folder_name)

    # for k in range(len(ground_truth_data_list)):
    #     secondary_training_data_onefile(ground_truth_data_list[k], reconstruction_folder_name)
    pool.starmap(secondary_training_data_onefile, zip(ground_truth_data_list, repeat(reconstruction_folder_name)))


