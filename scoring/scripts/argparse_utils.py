import os
from argparse import RawDescriptionHelpFormatter
import argparse
from utils import *


def add_args(parser):

     parser.add_argument("-target", type=str, default=None,
                         help="Input. This pdb code of target.")

     parser.add_argument("-pose_fpath", type=str, default=None,
                         help="Input. This dpath of poses.")

     parser.add_argument("-rec_fpath", type=str, default=None,
                         help="Input. pdb file. \n"
                              "This file contains the conformation of the protein.")

     parser.add_argument("-mean_std_file", type=str, default="train_mean_std.csv",
                         help="Input. csv file. \n"
                              "This file contains the mean and std values in training set.")

     parser.add_argument("-model", type=str, default="bestmodel.pth",
                         help="Input. pth file. \n"
                              "The final DL model.")

     parser.add_argument("-out_fpath", type=str, default="out_results.csv",
                         help="Output. This file contains the rmsd and vina score of receptor-ligand complex.")                        