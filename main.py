import argparse
import random
import torch
import numpy as np
from pathlib import Path

from datasets.dataloader import DataLoader
from model import Trainer
from utils.utils import create_folders
from batch_gen import BatchGenerator

from asformer import MyTransformer, ASFormerTrainer

def main(args, device, model_load_dir, model_save_dir, results_save_dir):
    if args.action == 'train' :
        # load train dataset and test dataset
        print(f'Load train data: {args.train_data}')
        train_loader = DataLoader(args, args.train_data, 'train')
        print(f'Load test data: {args.test_data}')
        test_loader = DataLoader(args, args.test_data, 'test')
        
        print(f'Start training.')
        trainer = ASFormerTrainer(
            args.num_layers,
            args.r1,
            args.r2, 
            args.num_f_maps,
            args.input_dim,
            train_loader.num_classes,
            args.channel_masking_rate,
            train_loader.weights, 
            model_save_dir)

        eval_args = [
            args,
            model_save_dir,
            results_save_dir,
            test_loader.features_dict,
            test_loader.gt_dict,
            test_loader.eval_gt_dict,
            test_loader.vid_list,
            args.num_epochs,
            device,
            'eval',
            args.classification_threshold,
        ]

        batch_gen = BatchGenerator(
            train_loader.num_classes,
            train_loader.gt_dict,
            train_loader.features_dict,
            train_loader.eval_gt_dict
            )

        batch_gen.read_data(train_loader.vid_list)
        
        trainer.train(model_save_dir,
                      batch_gen,
                      args.num_epochs, 
                      args.bz,
                      args.lr,
                      eval_args)

    else:
        print(f'Load test data: {args.test_data}')
        test_loader = DataLoader(args, args.test_data, args.extract_set, results_dir=results_save_dir)

        trainer = ASFormerTrainer(
            args.num_layers,
            args.r1,
            args.r2, 
            args.num_f_maps,
            args.input_dim,
            test_loader.num_classes,
            args.channel_masking_rate,
            test_loader.weights, 
            model_save_dir)

        trainer.test(
            args,
            model_load_dir,
            results_save_dir,
            test_loader.features_dict,
            test_loader.gt_dict,
            test_loader.eval_gt_dict,
            test_loader.vid_list,
            args.num_epochs,
            device,
            'test',
            args.classification_threshold,
            uniform=0,
            save_pslabels=False,
            CP_dict=test_loader.CP_dict,
            )

class Args(): 
    def __init__(self, *args, **kwargs):
        self.train_data = 'bslcp'
        self.test_data = 'bslcp'
        self.i3d_training = 'i3d_kinetics_bslcp_981'
        self.num_in_frames = 16
        self.features_dim = 1024
        self.weights = 'opt'
        self.regression = 0 
        self.feature_normalization = 0
        self.eval_use_CP = 0
        self.bz = 1
        self.action = 'train'
        self.seed = 0 
        self.refresh = 'store_true'

        ## Transformer : 
        self.nhead = 8
        self.nhid = 1024
        self.dim_feedforward = 1024
        self.num_layers = 6
        self.dropout = 0.3

        ## ASFormer : 
        self.num_layers = 7
        self.num_decoders = 3
        self.r1 = 2
        self.r2 = 2
        self.channel_masking_rate = 0.3
        self.input_dim = 1024
        self.num_f_maps = 64
        
        
        ## Optimization
        self.num_epochs = 20
        self.lr = 0.0005
        
        ## inference
        self.classification_threshold = 0.5
        self.folder = ''
        
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device}")
    args = Args()

    # set seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # create models and save args
    model_load_dir, model_save_dir, results_save_dir = create_folders(args)
    main(args, device, model_load_dir, model_save_dir, results_save_dir)
