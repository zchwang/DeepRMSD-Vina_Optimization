from spyrmsd import io, rmsd
import os
import pandas as pd
import numpy as np
import torch
from parse_ligand import Ligand

def output_pdb(traj_fpath: str,
               temp_fpath: str,
               ligand: Ligand):
    """Output the new conformation into the trajectory file.

    Args:
        epoch (int): Epoch number.
        output_path (str): output directory path.
        file_name (str): output trajectory file path.
        origin_heavy_atoms_lines (list): Original heavy atoms lines.
        new_coords (torch.Tensor): predicted ligand atom coordinates.
    """

    origin_heavy_atoms_lines = ligand.origin_heavy_atoms_lines
    new_coords = ligand.pose_heavy_atoms_coords

    lines = []
    for num, line in enumerate(origin_heavy_atoms_lines):
        x = new_coords[num][0].detach().numpy()
        y = new_coords[num][1].detach().numpy()
        z = new_coords[num][2].detach().numpy()

        atom_type = line.split()[2]
        pre_element = line.split()[2]
        if pre_element[:2] == "CL":
            element = "Cl"
        elif pre_element[:2] == "BR":
            element = "Br"
        else:
            element = pre_element[0]

        line = "ATOM%7s%5s%4s%2s%4s%12s%8s%8s%6s%6s%12s" % (
            str(num + 1), atom_type, "LIG", "A", "1", "%.3f" % x, "%.3f" % y, "%.3f" % z, "1.00", "0.00", element)
        lines.append(line)

    MODEL_numner = 0
    if not os.path.exists(traj_fpath):
        pass
    else:
        with open(traj_fpath) as f:
            MODEL_numner = len([x for x in f.readlines() if x[:5] == "MODEL"])

    with open(traj_fpath, 'a+') as f:

        f.writelines('MODEL%9s\n' % str(MODEL_numner + 1))

        for line in lines:
            f.writelines(line + '\n')
        f.writelines("TER\n")
        f.writelines("ENDMDL\n")

    with open(temp_fpath, 'w') as f:
        for line in lines:
            f.writelines(line + '\n')


def save_final_cnfr(final_cnfr_fpath,
                    ligand: Ligand):

    origin_heavy_atoms_lines = ligand.origin_heavy_atoms_lines
    new_coords = ligand.pose_heavy_atoms_coords

    lines = []
    for num, line in enumerate(origin_heavy_atoms_lines):
        x = new_coords[num][0].detach().numpy()
        y = new_coords[num][1].detach().numpy()
        z = new_coords[num][2].detach().numpy()

        atom_type = line.split()[2]
        pre_element = line.split()[2]
        if pre_element[:2] == "CL":
            element = "Cl"
        elif pre_element[:2] == "BR":
            element = "Br"
        else:
            element = pre_element[0]

        line = "ATOM%7s%5s%4s%2s%4s%12s%8s%8s%6s%6s%12s" % (
        str(num + 1), atom_type, "LIG", "A", "1", "%.3f" % x, "%.3f" % y, "%.3f" % z, "1.00", "0.00", element)
        lines.append(line)

    with open(final_cnfr_fpath, 'w') as f:
        for line in lines:
            f.writelines(line + '\n')


def save_data(preds_trues, name):

    preds_trues = np.array(preds_trues)
    index = [x for x in range(preds_trues.shape[0])]
    df = pd.DataFrame(preds_trues, index=index, columns=['vina', 'pred_rmsd', 'rmsd+vina', 'real_rmsd'])

    df.to_csv(name)

    return df


def save_results(file_, init, final):
    values = np.r_[init, final].reshape(2, 4)
    df = pd.DataFrame(values, index=['init', 'final'], columns=['vina', 'pred', 'rmsd+vina', 'true'])

    df.to_csv(file_)

def cal_hrmsd(ref_mol, test):
    try:
        ref = io.loadmol(ref_mol)
        ref.strip()

        mol = io.loadmol(test)
        mol.strip()

        coords_ref = ref.coordinates
        anum_ref = ref.atomicnums

        each_coords = mol.coordinates
        each_anum = mol.atomicnums

        RMSD = rmsd.hrmsd(coords_ref, each_coords, anum_ref, each_anum, center=False)

        return RMSD

    except Exception as e:
        print("Error:", e)


def set_lr(epoch, torsion_param, number_of_frames):

    lr_xyz = 0.05
    lr_rotation = 0.05
    lr_torsion = torsion_param / (number_of_frames + 1)

    slope_xyz = (lr_xyz - 0.01) / 50
    slope_rotation = (lr_rotation - 0.01) / 50
    slope_torsion = (lr_torsion - 0.05) / 50

    if epoch > 50:
        epoch = 50

    lr_xyz = lr_xyz - slope_xyz * epoch
    lr_rotation = lr_rotation - slope_rotation * epoch
    lr_torsion = lr_torsion - slope_torsion * epoch

    lr = torch.zeros(6 + number_of_frames)

    for i in range(3):
        lr[i] = lr_xyz

    for i in range(3, 6):
        lr[i] = lr_rotation

    for i in range(6, 6 + number_of_frames):
        lr[i] = lr_torsion

    return lr
