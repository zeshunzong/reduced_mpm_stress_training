import argparse
import os
import warnings
import glob
from SimulationDataModule import *
from CROMnet_secondary import *
from Callbacks import *
from util import *

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.strategies import DDPStrategy
from pprint import pprint


def main(args):
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    output_path = parent_dir + '/training_outputs'
    time_string = getTime()
    if args.mode == "train": time_string = "train_stress_from_" + os.path.normpath(args.d) + "_at_" + time_string
    weightdir = output_path + '/weights/' + time_string
    checkpoint_callback, lr_monitor, epoch_timer, custom_progress_bar = CustomCheckPointCallback(verbose=True, dirpath=weightdir, save_last=True), LearningRateMonitor(logging_interval='step'), EpochTimeCallback(), LitProgressBar()
    callbacks=[lr_monitor, checkpoint_callback, epoch_timer, custom_progress_bar]
    logdir = output_path + '/logs'
    logger = pl_loggers.TensorBoardLogger(save_dir=logdir, name='', version=time_string, log_graph=False)

    # sparse_trainer
    sparse_lr = [10.0, 5.0, 2.0, 1.0]
    sparse_epo= [1000, 1000, 500, 200]
    sparse_batch_size = 10
    sparse_skip = 2
    sparse_train_every_n_points = 10
    trainer = Trainer.from_argparse_args(args, gpus=findEmptyCudaDeviceList(args.gpus), default_root_dir=output_path, callbacks=callbacks, logger=logger, max_epochs= np.sum(sparse_epo), log_every_n_steps=10, strategy=DDPStrategy(find_unused_parameters=False))
    # fine trainer
    # fine_lr = [1.0, 0.2]
    # fine_epo= [30, 10]
    # fine_batch_size = 1

    if args.mode == "train":
        if args.d:
            data_path = args.d
            parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
            data_path = parent_dir + "/mpm-data/training_data/" + data_path
           
            dm = SimulationDataModule(data_path, sparse_batch_size, num_workers=24, train_every_k_timestep=sparse_skip, train_every_n_points=sparse_train_every_n_points)
            data_format, example_input_array = dm.get_dataFormat() # example_input_array has shape (1, n_particle, lbl + input_dim + mu_dim)
            preprop_params = dm.get_dataParams()

            network_kwargs = get_validArgs(CROMnet_secondary, args)
            label_length = data_format['lbl']
            network_kwargs['lbl'] = label_length
            network_kwargs['lr']  = sparse_lr
            network_kwargs['epo'] = sparse_epo
            network_kwargs['batch_size'] = sparse_batch_size
            network_kwargs['train_every_n_points'] = sparse_train_every_n_points
            
            net = CROMnet_secondary(data_format, preprop_params, example_input_array, **network_kwargs)
            trainer.fit(net, dm)

            ####################################### fine tuning ###########################
            # fine_trainer = Trainer.from_argparse_args(args, gpus=findEmptyCudaDeviceList(args.gpus), default_root_dir=output_path, callbacks=callbacks, logger=logger, max_epochs= np.sum(fine_epo), log_every_n_steps=10, strategy=DDPStrategy(find_unused_parameters=False))
            # fileList = glob.glob(os.getcwd() +"/../training_outputs/weights/" + time_string + "/*.ckpt")
            # checkpoint = CROMnet_secondary.load_from_checkpoint(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/training_outputs/weights/" + time_string + "/" + fileList[0].split(os.sep)[-1])
            # checkpoint.learning_rates, checkpoint.accumulated_epochs = generateEPOCHS(fine_lr, fine_epo)
            # checkpoint.batch_size = fine_batch_size
            # dm_fine = SimulationDataModule(data_path, fine_batch_size, num_workers=24)
            # data_format_fine, example_input_array_fine = dm_fine.get_dataFormat()
            # checkpoint.data_format = data_format_fine
            # fine_trainer.fit(checkpoint, dm_fine)
        
        else:
            exit('Enter data path')
    
    elif args.mode == "reconstruct":

        if args.m:

            weight_path = args.m
            parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            weight_path = parent_dir + '/training_outputs/weights/' + weight_path
            net = CROMnet_secondary.load_from_checkpoint(weight_path, loaded_from=weight_path)
            dm = SimulationDataModule(net.data_format['data_path'], 1, num_workers=24, train_every_k_timestep=1, train_every_n_points=1)

        else:
            exit('Enter weight path')

        trainer.test(net, dm)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Neural Representation training')

    # Mode for script
    parser.add_argument('-mode', help='train or test',
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

    # Trainer arguments
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    main(args)