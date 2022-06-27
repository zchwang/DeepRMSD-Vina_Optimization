import os
import torch
import numpy as np
import pandas as pd
from argparse import RawDescriptionHelpFormatter
import argparse
from utils import *



def add_args(parser):


    parser.add_argument("-output_path", type=str, default="./output_path",
                        help="Input. The path of output files.")

    parser.add_argument("-pose", type=str, default="decoy_pose.pdbqt",
                        help="Input. This file contains the conformational information \n"
                             "of decoys.")

    parser.add_argument("-native_pose", type=str, default="ligand.mol2",
                        help="Input. mol2 file. \n"
                             "This file contains the native conformation of the ligand.")

    parser.add_argument("-receptor", type=str, default="protein.pdb",
                        help="Input. pdb file. \n"
                             "This file contains the conformation of the protein.")

    parser.add_argument("-mean_std_file", type=str, default="train_mean_std.csv",
                        help="Input. csv file. \n"
                             "This file contains the mean and std values in training set.")

    parser.add_argument("-model", type=str, default="bestmodel.pth",
                        help="Input. pth file. \n"
                             "The final DL model.")

    
