from utils import *
import numpy as np
import pandas as pd
import torch
import json
import itertools
from parse_ligand import Ligand
from parse_receptor import Receptor
from model import CNN
import os, sys
import time

_current_dpath = os.path.dirname(os.path.abspath(__file__))

all_residues = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 'TRP', 'SER',
                'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 'GLU', 'LYS', 'ARG', 'HIS', 'OTH']
rec_elements = ['C', 'O', 'N', 'S', 'DU']
lig_elements = ['C', 'O', 'N', 'P', 'S', 'Hal', 'DU']

Hal = ['F', 'Cl', 'Br', 'I']


def get_residue(r_atom):
    r, a = r_atom.split('-')
    if not r in all_residues:
        r = 'OTH'
    if a in rec_elements:
        a = a
    else:
        a = 'DU'
    return r + '-' + a


def get_elementtype(e):
    if e in lig_elements:
        return e
    elif e in Hal:
        return 'Hal'
    else:
        return 'DU'


residues_atoms_pairs = ["-".join(x) for x in list(itertools.product(all_residues, rec_elements))]
keys = []
for r, a in list(itertools.product(residues_atoms_pairs, lig_elements)):
    keys.append('r6_' + r + '_' + a)
    keys.append('r1_' + r + '_' + a)

class ScoringFunction(object):

    def __init__(self,
                 receptor: Receptor = None,
                 ligand: Ligand = None,
                 mean_std_file: str = None,
                 model_fpath: str = None,
                 pre_cut: float = 0.3,
                 cutoff: float = 2.0,
                 n_features: int = 1470,
                 alpha: float = 0.5,
                 weight_1: float = 0.5,
                 ):

        # Parameters required for DeepRMSD
        self.repulsive_ = 6
        self.pre_cut = pre_cut  # Minimum interaction distance between protein-ligand atoms
        self.cutoff = cutoff  # Maximum interaction distance between protein-ligand atoms
        self.n_features = n_features  # Length of feature vector
        # self.shape = shape  # The dimension of the feature matrix input into the CNN model.
        self.mean_std_file = mean_std_file  # Record the mean and standard deviation of the feature values in the training set.
        self.model_fpath = model_fpath  # The path of the trained CNN model

        # The params of RMSD_Vina when calculating the combined score.
        self.alpha = alpha  # The weight of RMSD_Vina in the combined score.
        self.weight_1 = weight_1  # Weights when RMSD and Vina score are added.
        self.weight_2 = 1.0 - weight_1

        # ligand
        self.ligand = ligand
        self.pose_heavy_atoms_coords = self.ligand.pose_heavy_atoms_coords
        self.lig_heavy_atoms_element = self.ligand.lig_heavy_atoms_element
        self.updated_lig_heavy_atoms_xs_types = self.ligand.updated_lig_heavy_atoms_xs_types
        self.lig_root_atom_index = self.ligand.root_heavy_atom_index
        self.lig_frame_heavy_atoms_index_list = self.ligand.frame_heavy_atoms_index_list
        self.lig_torsion_bond_index = self.ligand.torsion_bond_index
        self.num_of_lig_ha = self.ligand.number_of_heavy_atoms
        self.number_of_poses = len(self.pose_heavy_atoms_coords)

        # receptor
        self.receptor = receptor
        self.rec_heavy_atoms_xyz = self.receptor.init_rec_heavy_atoms_xyz
        self.rec_heavy_atoms_xs_types = self.receptor.rec_heavy_atoms_xs_types
        self.residues_heavy_atoms_pairs = self.receptor.residues_heavy_atoms_pairs
        self.heavy_atoms_residues_indices = self.receptor.heavy_atoms_residues_indices
        self.rec_index_to_series_dict = self.receptor.rec_index_to_series_dict
        self.num_of_rec_ha = len(self.receptor.init_rec_heavy_atoms_xyz)

        self.rec_carbon_is_hydrophobic_dict = {}
        self.rec_atom_is_hbdonor_dict = {}

        # variable of the protein-ligand interaction
        self.dist = torch.tensor([])
        self.intra_repulsive_term = torch.tensor(1e-6)
        self.inter_repulsive_term = torch.tensor(1e-6)
        self.FR_repulsive_term = torch.tensor(1e-6)

        self.vina_inter_energy = 0.0

        self.origin_energy = torch.tensor([])  # the energy matrix before scaler
        self.features_matrix = torch.tensor([])  # the energy matrix after scaler

        self.pred_rmsd = torch.tensor([])

        # predefined parameters
        with open(os.path.join(_current_dpath, "atomtype_mapping.json")) as f:
            self.atomtype_mapping = json.load(f)

        with open(os.path.join(_current_dpath, "covalent_radii_dict.json")) as f:
            self.covalent_radii_dict = json.load(f)

        with open(os.path.join(_current_dpath, "vdw_radii_dict.json")) as f:
            self.vdw_radii_dict = json.load(f)
    
    def generate_pldist_mtrx(self):

        self.rec_heavy_atoms_xyz = self.rec_heavy_atoms_xyz.expand(len(self.pose_heavy_atoms_coords), -1, 3)

        # Generate the distance matrix of heavy atoms between the protein and the ligand.
        n, N, C = self.rec_heavy_atoms_xyz.size()
        n, M, _ = self.pose_heavy_atoms_coords.size()
        dist = -2 * torch.matmul(self.rec_heavy_atoms_xyz, self.pose_heavy_atoms_coords.permute(0, 2, 1))
        dist += torch.sum(self.rec_heavy_atoms_xyz ** 2, -1).view(-1, N, 1)
        dist += torch.sum(self.pose_heavy_atoms_coords ** 2, -1).view(-1, 1, M)

        dist = (dist >= 0) * dist
        self.dist = torch.sqrt(dist)

        return self

    def cal_RMSD(self):

        dist_nm = self.dist / 10
       
        # Generate the feature matrix for predict RMSD by DeepRMSD.
        #t = time.time()
        dist_nm_1 = (dist_nm <= self.pre_cut) * self.pre_cut
        dist_nm_2 = dist_nm * (dist_nm > self.pre_cut) * (dist_nm < self.cutoff)
        #print("cost time in cutoff dist:", time.time() - t)

        # r6-term
        #t = time.time()
        features_matrix_1 = torch.pow(dist_nm_1 + (dist_nm_1 == 0.) * 1., -6) - (dist_nm_1 == 0.) * 1.
        features_matrix_2 = torch.pow(dist_nm_2 + (dist_nm_2 == 0.) * 1., -6) - (dist_nm_2 == 0.) * 1.
        features_1 = (features_matrix_1 + features_matrix_2).reshape(-1, 1)
        #print("cost time in r6:", time.time() - t)

        # r1-term
        #t = time.time()
        features_matrix_1 = torch.pow(dist_nm_1 + (dist_nm_1 == 0.) * 1., -1) - (dist_nm_1 == 0.) * 1.
        features_matrix_2 = torch.pow(dist_nm_2 + (dist_nm_2 == 0.) * 1., -1) - (dist_nm_2 == 0.) * 1.
        features_2 = (features_matrix_1 + features_matrix_2).reshape(-1, 1)
        #print("cost time in r1:", time.time() - t)

        # Concatenate the r6 and r1 feature matrices together
        #t = time.time()
        features = torch.cat((features_1, features_2), axis=1)
        features = features.reshape(self.number_of_poses, 1, -1)
        #print("cost time in cat features:", time.time() - t)

        # atom type combination
        #t = time.time()
        residues_heavy_atoms_pairs = [get_residue(x) for x in self.residues_heavy_atoms_pairs]
        lig_heavy_atoms_element = [get_elementtype(x) for x in self.lig_heavy_atoms_element]
        #print("cost time in map res-atom types:", time.time() - t)

        #t = time.time()
        rec_lig_ele = ["_".join(x) for x in
                       list(itertools.product(residues_heavy_atoms_pairs, lig_heavy_atoms_element))]
        #print("cost time in res-atom pairs:", time.time() - t)

        #t = time.time()
        rec_lig_atoms_combines = []
        for i in rec_lig_ele:
            rec_lig_atoms_combines.append("r6_" + i)
            rec_lig_atoms_combines.append("r1_" + i)
        #print("cost time in rec-lig combine:", time.time() - t)

        # encode each atom pair type into a matrix
        # t = time.time()
        if not "init_matrix" in globals().keys():
            global init_matrix
        init_matrix = torch.zeros(len(rec_lig_atoms_combines), 1470)

        for num, c in enumerate(rec_lig_atoms_combines):
            key_num = keys.index(c)
            init_matrix[num][key_num] = 1
        #print("cost time in init matrix:", time.time() - t)

        init_matrix = init_matrix.expand(self.number_of_poses, init_matrix.shape[0], init_matrix.shape[1])

        # generate the final energy matrix
        #t = time.time()
        matrix = torch.matmul(features, init_matrix)
        self.origin_energy = matrix.reshape(-1, 1470)
        #print("cost time in get features:", time.time() - t)

        # Standardize features
        scaler = pd.read_csv(self.mean_std_file, index_col=0)
        means = torch.from_numpy(scaler.values[0, :].astype(np.float32))
        stds = (torch.from_numpy(scaler.values[1, :].astype(np.float32)) + 1e-6)

        #t = time.time()
        matrix = (self.origin_energy - means) / stds
        self.features_matrix = matrix
        #print("cost time in scaler features:", time.time() - t)

        # predict the RMSD
        model = torch.load(self.model_fpath)
        #t = time.time()
        self.pred_rmsd = model(self.features_matrix)
        #print("cost time in pred rmsd:", time.time() - t)

        return self

    def cal_inter_repulsion(self, dist, vdw_sum):

        """
             When the distance between two atoms from the protein-ligand complex is less than the sum of the van der Waals radii,
            an intermolecular repulsion term is generated.
        """
        _cond = (dist < vdw_sum) * 1.
        _cond_sum = torch.sum(_cond, axis=1)
        _zero_indices = torch.where(_cond_sum == 0)[0]
        for index in _zero_indices:
            index = int(index)
            _cond[index][0] = torch.pow(dist[index][0], 20)

        self.inter_repulsive_term = torch.sum(torch.pow(_cond * dist + (_cond * dist == 0) * 1., -1 * self.repulsive_), axis=1) - \
                torch.sum((_cond * dist) * 1., axis=1)
        
        self.inter_repulsive_term = self.inter_repulsive_term.reshape(-1, 1)

        return self

    def cal_intra_repulsion(self):

        """

            When the distance between two atoms in adjacent frames are less than the sum of the van der Waals radii
        of the two atoms, an intramolecular repulsion term is generated.

        """
        all_root_frame_heavy_atoms_index_list = [self.lig_root_atom_index] + self.lig_frame_heavy_atoms_index_list
        number_of_all_frames = len(all_root_frame_heavy_atoms_index_list)

        dist_list = []
        vdw_list = []

        for frame_i in range(0, number_of_all_frames - 1):
            for frame_j in range(frame_i + 1, number_of_all_frames):

                for i in all_root_frame_heavy_atoms_index_list[frame_i]:
                    for j in all_root_frame_heavy_atoms_index_list[frame_j]:

                        if [i, j] in self.lig_torsion_bond_index or [j, i] in self.lig_torsion_bond_index:
                            continue

                        # angstrom
                        d = torch.sqrt(
                            torch.sum(
                                torch.square(self.pose_heavy_atoms_coords[:, i] - self.pose_heavy_atoms_coords[:, j]),
                                axis=1))
                        dist_list.append(d.reshape(-1, 1))

                        i_xs = self.updated_lig_heavy_atoms_xs_types[i]
                        j_xs = self.updated_lig_heavy_atoms_xs_types[j]

                        # angstrom
                        vdw_distance = self.vdw_radii_dict[i_xs] + self.vdw_radii_dict[j_xs]
                        vdw_list.append(torch.tensor([vdw_distance]))
    
        dist_tensor = torch.cat(dist_list, axis=1)
        vdw_tensor = torch.cat(vdw_list, axis=0)
    
        self.intra_repulsive_term = torch.sum(torch.pow((dist_tensor < vdw_tensor) * 1. * dist_tensor + \
                                                (dist_tensor >= vdw_tensor) * 1., -1 * self.repulsive_), axis=1) - \
                                                torch.sum((dist_tensor >= vdw_tensor) * 1., axis=1)

        self.intra_repulsive_term = self.intra_repulsive_term.reshape(-1, 1)

        return self

    def get_vdw_radii(self, xs):
        return self.vdw_radii_dict[xs]

    def get_vina_dist(self, r_index, l_index):
        return self.dist[:, r_index, l_index]

    def get_vina_rec_xs(self, index):
        return self.rec_heavy_atoms_xs_types[index]

    def get_vina_lig_xs(self, index):
        return self.updated_lig_heavy_atoms_xs_types[index]

    def is_hydrophobic(self, index, is_lig):

        if is_lig == True:
            atom_xs = self.updated_lig_heavy_atoms_xs_types[index]
        else:
            atom_xs = self.rec_heavy_atoms_xs_types[index]

        return atom_xs in ["C_H", "F_H", "Cl_H", "Br_H", "I_H"]

    def is_hbdonor(self, index, is_lig):

        if is_lig == True:
            atom_xs = self.updated_lig_heavy_atoms_xs_types[index]
        else:
            atom_xs = self.rec_heavy_atoms_xs_types[index]

        return atom_xs in ["N_D", "N_DA", "O_DA", "Met_D"]

    def is_hbacceptor(self, index, is_lig):

        if is_lig == True:
            atom_xs = self.updated_lig_heavy_atoms_xs_types[index]
        else:
            atom_xs = self.rec_heavy_atoms_xs_types[index]

        return atom_xs in ["N_A", "N_DA", "O_A", "O_DA"]

    def is_hbond(self, atom_1, atom_2):
        return (
                (self.is_hbdonor(atom_1) and self.is_hbacceptor(atom_2)) or
                (self.is_hbdonor(atom_2) and self.is_hbacceptor(atom_1))
        )

    def _pad(self, vector, _Max_dim):

        _vec = torch.zeros(_Max_dim - len(vector))
        new_vector = torch.cat((vector, _vec), axis=0)

        return new_vector

    def cal_vina(self):

        rec_atom_indices_list = []  # [[]]
        lig_atom_indices_list = []  # [[]]
        all_selected_rec_atom_indices = []
        all_selected_lig_atom_indices = []

        _Max_dim = 0
        for each_dist in self.dist:

            each_rec_atom_indices, each_lig_atom_indices = torch.where(each_dist <= 8)
            rec_atom_indices_list.append(each_rec_atom_indices.numpy().tolist())
            lig_atom_indices_list.append(each_lig_atom_indices.numpy().tolist())
            all_selected_rec_atom_indices += each_rec_atom_indices.numpy().tolist()
            all_selected_lig_atom_indices += each_lig_atom_indices.numpy().tolist()

            if len(each_rec_atom_indices) > _Max_dim:
                _Max_dim = len(each_rec_atom_indices)

        all_selected_rec_atom_indices = list(set(all_selected_rec_atom_indices))
        all_selected_lig_atom_indices = list(set(all_selected_lig_atom_indices))

        # Update the xs atom type of heavy atoms for receptor.
        t = time.time()
        for i in all_selected_rec_atom_indices:
            i = int(i)
            self.receptor.update_rec_xs(self.rec_heavy_atoms_xs_types[i], i,
                                        self.rec_index_to_series_dict[i], self.heavy_atoms_residues_indices[i])
        #print("cost time in update xs:", time.time() - t)

        # is_hydrophobic
        rec_atom_is_hydrophobic_dict = dict(zip(all_selected_rec_atom_indices,
                                                np.array(list(map(self.is_hydrophobic, all_selected_rec_atom_indices,
                                                                  [False] * len(all_selected_rec_atom_indices)))) * 1.))
        lig_atom_is_hydrophobic_dict = dict(zip(all_selected_lig_atom_indices,
                                                np.array(list(map(self.is_hydrophobic, all_selected_lig_atom_indices,
                                                                  [True] * len(all_selected_lig_atom_indices)))) * 1.))

        # is_hbdonor
        rec_atom_is_hbdonor_dict = dict(zip(all_selected_rec_atom_indices,
                                            np.array(list(map(self.is_hbdonor, all_selected_rec_atom_indices,
                                                              [False] * len(all_selected_rec_atom_indices)))) * 1.))
        lig_atom_is_hbdonor_dict = dict(zip(all_selected_lig_atom_indices,
                                            np.array(list(map(self.is_hbdonor, all_selected_lig_atom_indices,
                                                              [True] * len(all_selected_lig_atom_indices)))) * 1.))

        # is_hbacceptor
        rec_atom_is_hbacceptor_dict = dict(zip(all_selected_rec_atom_indices,
                                               np.array(list(map(self.is_hbacceptor, all_selected_rec_atom_indices,
                                                                 [False] * len(all_selected_rec_atom_indices)))) * 1.))
        lig_atom_is_hbacceptor_dict = dict(zip(all_selected_lig_atom_indices,
                                               np.array(list(map(self.is_hbacceptor, all_selected_lig_atom_indices,
                                                                 [True] * len(all_selected_lig_atom_indices)))) * 1.))

        rec_lig_is_hydrophobic = []
        rec_lig_is_hbond = []
        rec_lig_atom_vdw_sum = []
        for each_rec_indices, each_lig_indices in zip(rec_atom_indices_list, lig_atom_indices_list):

            r_hydro = []
            l_hydro = []
            r_hbdonor = []
            l_hbdonor = []
            r_hbacceptor = []
            l_hbacceptor = []

            r_vdw = []
            l_vdw = []

            for r_index, l_index in zip(each_rec_indices, each_lig_indices):
                # is hydrophobic
                r_hydro.append(rec_atom_is_hydrophobic_dict[r_index])
                l_hydro.append(lig_atom_is_hydrophobic_dict[l_index])

                # is hbdonor & hbacceptor
                r_hbdonor.append(rec_atom_is_hbdonor_dict[r_index])
                l_hbdonor.append(lig_atom_is_hbdonor_dict[l_index])

                r_hbacceptor.append(rec_atom_is_hbacceptor_dict[r_index])
                l_hbacceptor.append(lig_atom_is_hbacceptor_dict[l_index])

                # vdw 
                r_vdw.append(self.vdw_radii_dict[self.rec_heavy_atoms_xs_types[r_index]])
                l_vdw.append(self.vdw_radii_dict[self.updated_lig_heavy_atoms_xs_types[l_index]])

            # rec-atom hydro
            r_hydro = self._pad(torch.from_numpy(np.array(r_hydro)), _Max_dim)
            l_hydro = self._pad(torch.from_numpy(np.array(l_hydro)), _Max_dim)
            rec_lig_is_hydrophobic.append(r_hydro * l_hydro.reshape(1, -1))

            # hbond
            r_hbdonor = self._pad(torch.from_numpy(np.array(r_hbdonor)), _Max_dim)
            l_hbdonor = self._pad(torch.from_numpy(np.array(l_hbdonor)), _Max_dim)

            r_hbacceptor = self._pad(torch.from_numpy(np.array(r_hbacceptor)), _Max_dim)
            l_hbacceptor = self._pad(torch.from_numpy(np.array(l_hbacceptor)), _Max_dim)
            _is_hbond = ((r_hbdonor * l_hbacceptor + r_hbacceptor * l_hbdonor) > 0) * 1.
            rec_lig_is_hbond.append(_is_hbond.reshape(1, -1))

            # rec-lig vdw 
            rec_lig_atom_vdw_sum.append(
                self._pad(torch.from_numpy(np.array(r_vdw) + np.array(l_vdw)), _Max_dim).reshape(1, -1))

        rec_lig_is_hydrophobic = torch.cat(rec_lig_is_hydrophobic, axis=0)
        rec_lig_is_hbond = torch.cat(rec_lig_is_hbond, axis=0)

        rec_lig_atom_vdw_sum = torch.cat(rec_lig_atom_vdw_sum, axis=0)

        # vina dist 
        vina_dist_list = []

        for _num, dist in enumerate(self.dist):
            dist = dist * ((dist <= 8) * 1.)
            l = len(dist[dist != 0])
            vina_dist_list.append(self._pad(dist[dist != 0], _Max_dim).reshape(1, -1))

        vina_dist = torch.cat(vina_dist_list, axis=0)

        vina = VinaScoreCore(vina_dist, rec_lig_is_hydrophobic, rec_lig_is_hbond, rec_lig_atom_vdw_sum)
        vina_inter_term = vina.process()
        self.vina_inter_energy = vina_inter_term / (
                    1 + 0.05846 * (self.ligand.active_torsion + 0.5 * self.ligand.inactive_torsion))

        self.vina_inter_energy = self.vina_inter_energy.reshape(-1, 1)

        # inter clash
        self.cal_inter_repulsion(vina_dist, rec_lig_atom_vdw_sum)

        return self

    def process(self):
        t = time.time()
        self.generate_pldist_mtrx()
        #print("cost time in generate pldist:", time.time() - t)  # 0.41s

        t = time.time()
        self.cal_RMSD()
        #print("cost time in cal RMSD:", time.time() - t)  # 21.13s

        t = time.time()
        self.cal_vina()
        #print("cost time in cal vina:", time.time() - t)

        t = time.time()
        if self.ligand.number_of_frames == 0:
            self.intra_repulsive_term = 0
        else:
            self.cal_intra_repulsion()
        #print("cost time in cal intra:", time.time() - t)

        rmsd_vina = self.weight_1 * self.pred_rmsd + self.weight_2 * self.vina_inter_energy

        if self.ligand.number_of_frames == 0:
            combined_score = self.alpha * rmsd_vina + ((1 - self.alpha) / 2) * torch.log(self.inter_repulsive_term)
     
        else:
            combined_score = self.alpha * rmsd_vina + ((1 - self.alpha) / 2) * (
                torch.log(self.intra_repulsive_term) + torch.log(self.inter_repulsive_term))
    
       

        return self.pred_rmsd, self.vina_inter_energy, rmsd_vina, combined_score

class VinaScoreCore(object):

    def __init__(self, dist_matrix, rec_lig_is_hydrophobic, rec_lig_is_hbond, rec_lig_atom_vdw_sum):
        """
        Args:
            dist_matrix [N, M]: the distance matrix with less than 8 angstroms. N is the number of poses,
        M is the number of rec-lig atom pairs less than 8 Angstroms in each pose.

        Returns:
            final_inter_score [N, 1]

        """

        self.dist_matrix = dist_matrix
        self.rec_lig_is_hydro = rec_lig_is_hydrophobic
        self.rec_lig_is_hb = rec_lig_is_hbond
        self.rec_lig_atom_vdw_sum = rec_lig_atom_vdw_sum

    def score_function(self):
        d_ij = self.dist_matrix - self.rec_lig_atom_vdw_sum

        Gauss_1 = torch.sum(torch.exp(- torch.pow(d_ij / 0.5, 2)), axis=1) - torch.sum((d_ij == 0) * 1., axis=1)
        Gauss_2 = torch.sum(torch.exp(- torch.pow((d_ij - 3) / 2, 2)), axis=1) - \
                  torch.sum((d_ij == 0) * 1. * torch.exp(torch.tensor(-1 * 9 / 4)), axis=1)
        #print("Gauss_1:", Gauss_1)
        #print("Gauss_2:", Gauss_2)

        # Repulsion
        Repulsion = torch.sum(torch.pow(((d_ij < 0) * d_ij), 2), axis=1)
        #print("Repulsion:", Repulsion)

        # Hydrophobic
        Hydro_1 = self.rec_lig_is_hydro * (d_ij <= 0.5) * 1.

        Hydro_2_condition = self.rec_lig_is_hydro * (d_ij > 0.5) * (d_ij < 1.5) * 1.
        Hydro_2 = 1.5 * Hydro_2_condition - Hydro_2_condition * d_ij

        Hydrophobic = torch.sum(Hydro_1 + Hydro_2, axis=1)
        #print("Hydro:", Hydrophobic)

        # HBonding
        hbond_1 = self.rec_lig_is_hb * (d_ij <= -0.7) * 1.
        hbond_2 = self.rec_lig_is_hb * (d_ij < 0) * (d_ij > -0.7) * 1.0 * (- d_ij) / 0.7
        HBonding = torch.sum(hbond_1 + hbond_2, axis=1)
        #print("HB:", HBonding)

        inter_energy = - 0.035579 * Gauss_1 - 0.005156 * Gauss_2 + 0.840245 * Repulsion - 0.035069 * Hydrophobic - 0.587439 * HBonding

        return inter_energy

    def process(self):
        final_inter_score = self.score_function()

        return final_inter_score
