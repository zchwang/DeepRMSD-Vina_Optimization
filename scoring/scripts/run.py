from argparse import RawDescriptionHelpFormatter
import argparse
from re import L
from statistics import mode
from turtle import pos
from argparse_utils import add_args
import datetime
from model import CNN
from parse_receptor import Receptor
from parse_ligand import Ligand
from scoring_function import ScoringFunction
import numpy as np
import pandas as pd
import os

d = """

    Score the protein-ligand binding poses with DeepRMSD+Vina.

    """

parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
add_args(parser)
args = parser.parse_args()

target=args.target
rec_fpath = args.rec_fpath
pose_fpath = args.pose_fpath
out_fpath = args.out_fpath
mean_std_file = args.mean_std_file
model = args.model

#if not os.path.exists("results/" + target):
#    os.makedirs("results/" + target)


def perform_scoring(ligand: Ligand = None,
                    receptor: Receptor = None,
                    mean_std_file: str = None,
                    model_fpath: str = None):

    scoring = ScoringFunction(receptor=receptor, ligand=ligand, mean_std_file=mean_std_file, model_fpath=model_fpath)
    scoring.generate_pldist_mtrx()
    
    scoring.cal_RMSD()
    scoring.cal_vina()

    pred_rmsd = scoring.pred_rmsd.detach().numpy()
    inter_vina = scoring.vina_inter_energy.detach().numpy()

    rmsd_vina = 0.5 * pred_rmsd + 0.5 * inter_vina

    value = np.c_[pred_rmsd, inter_vina, rmsd_vina]
    value = np.round(value, 4)
    poses_names = ligand.poses_file_names

    df = pd.DataFrame(value, index=poses_names, columns=["pred_rmsd", "inter_vina", "rmsd_vina"])
    df = df.sort_values(by="rmsd_vina", ascending=True)
    df.to_csv(out_fpath)



ligand = Ligand(poses_file=pose_fpath)
ligand.parse_ligand()

receptor = Receptor(receptor_fpath=rec_fpath)
receptor.parse_receptor()

perform_scoring(ligand=ligand,
                receptor=receptor,
                mean_std_file=mean_std_file,
                model_fpath=model)
