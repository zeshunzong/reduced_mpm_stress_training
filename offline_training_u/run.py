import argparse
import os
import warnings
import sys
from SimulationDataModule import *
from CROMnet import *
from Callbacks import *
from util import *

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies import DDPStrategy



def main(args):

    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    output_path = parent_dir + '/training_outputs'
    time_string = getTime()
    if args.mode == "train": 
        time_string = "train_q" + "_at_" + time_string
        if args.d:
            mylist = os.path.normpath(args.d).split(os.path.sep)
            time_string = mylist[0] + "_" + time_string # mylist[0] is testcase name
    weightdir = output_path + '/weights/' + time_string
    checkpoint_callback, lr_monitor, epoch_timer, custom_progress_bar = CustomCheckPointCallback(verbose=True, dirpath=weightdir, save_last=True), LearningRateMonitor(logging_interval='step'), EpochTimeCallback(), LitProgressBar()
    callbacks=[lr_monitor, checkpoint_callback, epoch_timer, custom_progress_bar]
    logdir = output_path + '/logs'
    logger = pl_loggers.TensorBoardLogger(save_dir=logdir, name='', version=time_string, log_graph=False)

    sparse_lr = [10.0, 5.0, 2.0, 1.0]
    sparse_epo= [400, 200, 100, 50]
    # sparse_epo= [1, 1, 1, 1]
    sparse_batch_size = 10
    sparse_skip = 2
    sparse_train_every_n_points = 10
    trainer = Trainer.from_argparse_args(args, gpus=findEmptyCudaDeviceList(args.gpus), default_root_dir=output_path, callbacks=callbacks, logger=logger, max_epochs= np.sum(sparse_epo), log_every_n_steps=10, strategy=DDPStrategy(find_unused_parameters=False))


    if args.mode == "train":
        if args.d:
            data_path = args.d
            parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
            data_path = parent_dir + "/mpm-data/training_data/" + data_path
            
            dm = SimulationDataModule(data_path, sparse_batch_size, num_workers=24, train_every_k_timestep=sparse_skip, train_every_n_points=sparse_train_every_n_points)
            data_format, example_input_array = dm.get_dataFormat() # example_input_array has shape (1, n_particle, lbl + input_dim + mu_dim)
            preprop_params = dm.get_dataParams()

            network_kwargs = get_validArgs(CROMnet, args)
            network_kwargs['lr']  = sparse_lr
            network_kwargs['epo'] = sparse_epo
            network_kwargs['batch_size'] = sparse_batch_size
            network_kwargs['train_every_n_points'] = sparse_train_every_n_points
            net = CROMnet(data_format, preprop_params, example_input_array, **network_kwargs)
            net.train_every_n_points = sparse_train_every_n_points
            trainer.fit(net, dm)
          
        else:
            exit('Enter data path')
     
    
    elif args.mode == "reconstruct":

        if args.m:

            weight_path = args.m
            parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            weight_path = parent_dir + '/training_outputs/weights/' + weight_path
        
            net = CROMnet.load_from_checkpoint(weight_path, loaded_from=weight_path)
            # net.learning_rates, net.accumulated_epochs = generateEPOCHS([0.1], [1])
            
            # dm = SimulationDataModule(net.data_format['data_path'], net.batch_size, num_workers=64)
            dm = SimulationDataModule(net.data_format['data_path'], sparse_batch_size, num_workers=24, train_every_k_timestep=1, train_every_n_points=sparse_train_every_n_points)
            # trainer.fit(net, dm)
            # input("finish export")

        else:
            exit('Enter weight path')

        trainer.test(net, dm)

        if args.append_xhat_to_training == 1:
            folder = weight_path.split(os.sep)# os.path.split(weight_path)
            pred_folder = 'pred_' + folder[-2]

            ground_truth_folder = folder[-2][:-27]
            # print(ground_truth_folder)
            pred_folder = '../../mpm-data/training_data/' + pred_folder
            ground_truth_folder = '../../mpm-data/training_data/' + ground_truth_folder

            ground_truth_data_list = extract_h5_files(ground_truth_folder)

            from multiprocessing import Pool
            from functools import partial
            from itertools import repeat
            from multiprocessing import Pool, freeze_support
            pool = Pool(os.cpu_count())
            pool.starmap(secondary_training_data_onefile, zip(ground_truth_data_list, repeat(pred_folder)))

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
   


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Neural Representation training')

    # Mode for script
    parser.add_argument('-mode', help='train or reconstruct',
                    type=str, required=True)
    
    # Network arguments
    parser.add_argument('-lbl', help='label length',
                    type=int, required=False, default=6)  
    parser.add_argument('-scale_mlp', help='scale mlp',
                    type=int, required=False, default=10)
    parser.add_argument('-ks', help='scale mlp',
                    type=int, required=False, default=6)
    parser.add_argument('-strides', help='scale mlp',
                    type=int, required=False, default=4)
    parser.add_argument('-siren_dec', help='use siren - decoder',
                        action='store_true')
    parser.add_argument('-siren_enc', help='use siren - encoder',
                        action='store_true')                  
    parser.add_argument('-dec_omega_0', help='dec_omega_0',
                    type=float, required=False, default=30)
    parser.add_argument('-enc_omega_0', help='enc_omega_0',
                        type=float, required=False, default=0.3)
    
    # Network Training arguments
    parser.add_argument('-m', help='path to weight',
                    type=str, required=False)
    parser.add_argument('-d', help='path to the dataset',
                    type=str, required=False)
    parser.add_argument('-verbose', help='verbose',
                        action='store_false')
    parser.add_argument('-initial_lr', help='initial learning rate',
                        type=float, nargs=1, required=False, default=1e-4)
    parser.add_argument('-lr', help='adaptive learning rates',
                    type=float, nargs='*', required=False)
    parser.add_argument('-epo', help='adaptive epoch sizes',
                        type=int, nargs='*', required=False)
    parser.add_argument('-batch_size', help='batch size',
                    type=int, required=False, default=16)

    parser.add_argument('-skip_every', help='choose one frame for every k frames',
                    type=int, required=False, default=1)

    parser.add_argument('-append_xhat_to_training', help='if 1, then automatically add the computed label to ground truth (training data)',
                    type=int, required=False, default=0)

    # Trainer arguments
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)