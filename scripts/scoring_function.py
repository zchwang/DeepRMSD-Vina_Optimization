import numpy as np
import pandas as pd
import torch
import json
import itertools
import os
from parse_ligand import Ligand
from parse_receptor import Receptor

path = os.path.realpath(__file__)
dir_ = "/".join(path.split('/')[:-1])

class ScoringFunction(object):

    def __init__(self,
                 step: int = 0,
                 receptor: Receptor = None,
                 ligand: Ligand = None,
                 mean_std_file: str = None,
                 model_fpath: str = None,
                 pre_cut: float = 0.3,
                 cutoff: float = 2.0,
                 n_features: int = 1470,
          #       shape: tuple = (-1, 1, 210, 7),
                 alpha: float = 0.4,
                 weight_1: float = 0.4,
                 weight_2 : float = 0.6,
                 ):

        self.step = step

        # Parameters required for DeepRMSD
        self.repulsive_ = 12
        self.pre_cut = pre_cut  # Minimum interaction distance between protein-ligand atoms
        self.cutoff = cutoff  # Maximum interaction distance between protein-ligand atoms
        self.n_features = n_features  # Length of feature vector
        #self.shape = shape  # The dimension of the feature matrix input into the CNN model.
        self.mean_std_file = mean_std_file  # Record the mean and standard deviation of the feature values in the training set.
        self.model_fpath = model_fpath  # The path of the trained CNN model

        # The params of RMSD_Vina when calculating the combined score.
        self.alpha = alpha  # The weight of RMSD_Vina in the combined score.
        self.weight_1, self.weight_2 = weight_1, weight_2  # Weights when RMSD and Vina score are added.

        # ligand
        self.ligand = ligand
        self.pose_heavy_atoms_coords = self.ligand.pose_heavy_atoms_coords
        self.lig_heavy_atoms_element = self.ligand.lig_heavy_atoms_element
        self.updated_lig_heavy_atoms_xs_types = self.ligand.updated_lig_heavy_atoms_xs_types
        self.lig_root_atom_index = self.ligand.root_heavy_atom_index
        self.lig_frame_heavy_atoms_index_list = self.ligand.frame_heavy_atoms_index_list
        self.lig_torsion_bond_index = self.ligand.torsion_bond_index

        # receptor
        self.receptor = receptor
        self.rec_heavy_atoms_xyz = self.receptor.rec_heavy_atoms_xyz
        self.rec_heavy_atoms_xs_types = self.receptor.rec_heavy_atoms_xs_types
        self.residues_heavy_atoms_pairs = self.receptor.residues_heavy_atoms_pairs
        self.heavy_atoms_residues_indices = self.receptor.heavy_atoms_residues_indices
        self.rec_index_to_series_dict = self.receptor.rec_index_to_series_dict
        self.rec_carbon_is_hydrophobic_dict = {}
        self.rec_atom_is_hbdonor_dict = {}

        # variable of the protein-ligand interaction
        self.dist = torch.tensor([])
        self.intra_repulsive_term = torch.tensor(1e-6)
        self.inter_repulsive_term = torch.tensor(1e-6)

        self.vina_inter_energy = 0.0

        self.origin_energy = torch.tensor([])  # the energy matrix before scaler
        self.features_matrix = torch.tensor([])  # the energy matrix after scaler

        self.pred_rmsd = torch.tensor([])

        # predefined parameters
        with open(dir_ + "/atomtype_mapping.json") as f:
            self.atomtype_mapping = json.load(f)

        with open(dir_ + "/covalent_radii_dict.json") as f:
            self.covalent_radii_dict = json.load(f)

        with open(dir_ + "/vdw_radii_dict.json") as f:
            self.vdw_radii_dict = json.load(f)

    def cal_intra_repulsion(self):

        """

        When the distance between two atoms in adjacent frames are less than the sum of the van der Waals radii
        of the two atoms, an intramolecular repulsion term is generated.

        """

        all_root_frame_heavy_atoms_index_list = [self.lig_root_atom_index] + self.lig_frame_heavy_atoms_index_list
        number_of_all_frames = len(all_root_frame_heavy_atoms_index_list)

        for frame_i in range(0, number_of_all_frames - 1):
            for frame_j in range(frame_i + 1, number_of_all_frames):

                for i in all_root_frame_heavy_atoms_index_list[frame_i]:
                    for j in all_root_frame_heavy_atoms_index_list[frame_j]:

                        if [i, j] in self.lig_torsion_bond_index or [j, i] in self.lig_torsion_bond_index:
                            continue

                        # angstrom
                        d = torch.sqrt(
                            torch.sum(torch.square(self.pose_heavy_atoms_coords[i] - self.pose_heavy_atoms_coords[j])))

                        i_xs = self.updated_lig_heavy_atoms_xs_types[i]
                        j_xs = self.updated_lig_heavy_atoms_xs_types[j]

                        # angstrom
                        vdw_distance = self.vdw_radii_dict[i_xs] + self.vdw_radii_dict[j_xs]
                        if d <= vdw_distance:
                            self.intra_repulsive_term += 1. / torch.pow(d / 10, self.repulsive_)
        return self

    def cal_RMSD(self):

        # Generate the distance matrix of heavy atoms between the protein and the ligand.
        N, C = self.rec_heavy_atoms_xyz.size()
        M, _ = self.pose_heavy_atoms_coords.size()

        dist = -2 * torch.matmul(self.rec_heavy_atoms_xyz, self.pose_heavy_atoms_coords.permute(1, 0))
        dist += torch.sum(self.rec_heavy_atoms_xyz ** 2, -1).view(N, 1)
        dist += torch.sum(self.pose_heavy_atoms_coords ** 2, -1).view(1, M)

        self.dist = torch.sqrt(dist)
        dist_nm = self.dist / 10

        # Generate the feature matrix for predict RMSD by DeepRMSD.
        dist_nm_1 = (dist_nm <= self.pre_cut) * self.pre_cut
        dist_nm_2 = dist_nm * (dist_nm > self.pre_cut) * (dist_nm < self.cutoff)

        # r6-term
        features_matrix_1 = torch.pow(dist_nm_1 + (dist_nm_1 == 0.) * 1., -6) - (dist_nm_1 == 0.) * 1.
        features_matrix_2 = torch.pow(dist_nm_2 + (dist_nm_2 == 0.) * 1., -6) - (dist_nm_2 == 0.) * 1.
        features_1 = features_matrix_1 + features_matrix_2

        # r1-term
        features_matrix_1 = torch.pow(dist_nm_1 + (dist_nm_1 == 0.) * 1., -1) - (dist_nm_1 == 0.) * 1.
        features_matrix_2 = torch.pow(dist_nm_2 + (dist_nm_2 == 0.) * 1., -1) - (dist_nm_2 == 0.) * 1.
        features_2 = features_matrix_1 + features_matrix_2

        # Concatenate the r6 and r1 feature matrices together
        features = torch.cat((features_1.reshape(-1, 1), features_2.reshape(-1, 1)), 1)
        features = features.reshape(-1).reshape(1, -1)

        # atom type combination
        residues_heavy_atoms_pairs = [get_residue(x) for x in self.residues_heavy_atoms_pairs]
        lig_heavy_atoms_element = [get_elementtype(x) for x in self.lig_heavy_atoms_element]

        rec_lig_ele = ["_".join(x) for x in
                       list(itertools.product(residues_heavy_atoms_pairs, lig_heavy_atoms_element))]

        rec_lig_atoms_combines = []
        for i in rec_lig_ele:
            rec_lig_atoms_combines.append("r6_" + i)
            rec_lig_atoms_combines.append("r1_" + i)

        # encode each atom pair type into a matrix
        if self.step == 0:
            global init_matrix
            init_matrix = torch.zeros((len(rec_lig_atoms_combines), 1470))

            for num, c in enumerate(rec_lig_atoms_combines):
                key_num = keys.index(c)
                init_matrix[num][key_num] = 1

        # generate the final energy matrix
        matrix = torch.mm(features, init_matrix)
        self.origin_energy = matrix

        # Standardize features
        scaler = pd.read_csv(self.mean_std_file, index_col=0)
        means = torch.from_numpy(scaler.values[0, :].astype(np.float32))
        stds = torch.from_numpy(scaler.values[1, :].astype(np.float32)) + 1e-6

        matrix = (matrix - means) / stds
        self.features_matrix = matrix

        # predict the RMSD
        model = torch.load(self.model_fpath)
        self.pred_rmsd = model(self.features_matrix)

        return self

    def get_vdw_radii(self, xs):
        return self.vdw_radii_dict[xs]

    def inter_distance(self, r_xs_list, l_xs_list, dist):
        """
             When the distance between two atoms from the protein-ligand complex is less than the sum of the van der Waals radii,
            an intermolecular repulsion term is generated.
        """
        vdw_sum = torch.from_numpy(
            np.array(list(map(self.get_vdw_radii, r_xs_list))) + np.array(list(map(self.get_vdw_radii, l_xs_list))))

        self.inter_repulsive_term = torch.sum(
            1 / torch.pow(((dist <= vdw_sum) * dist + (dist > vdw_sum) * 10) / 10, self.repulsive_)) - torch.sum(
            (dist > vdw_sum) * 1.)

        return self

    def get_vina_dist(self, r_index, l_index):
        return self.dist[r_index, l_index]

    def get_vina_rec_xs(self, index):
        # print(index)
        return self.rec_heavy_atoms_xs_types[index]

    def get_vina_lig_xs(self, index):
        return self.updated_lig_heavy_atoms_xs_types[index]

    def cal_vina(self):

        """
        Calculate the intermolecular energy of AutoDock Vina.
        """

        # Get the index of atom pairs within 8 Angstroms.
        rec_atom_indices, lig_atom_indices = torch.where(self.dist <= 8)

        residue_index = []
        current_r_xs = []
        previous_series = []

        for i in rec_atom_indices:
            i = int(i)
            residue_index.append(self.heavy_atoms_residues_indices[i])
            current_r_xs.append(self.rec_heavy_atoms_xs_types[i])
            previous_series.append(self.rec_index_to_series_dict[i])

        # Update the xs atom type of heavy atoms for receptor.
        for i in range(len(current_r_xs)):
            self.receptor.update_rec_xs(current_r_xs[i], rec_atom_indices[i], previous_series[i], residue_index[i])

        # Get xs types
        vina_rec_xs = list(map(self.get_vina_rec_xs, rec_atom_indices))
        vina_lig_xs = list(map(self.get_vina_lig_xs, lig_atom_indices))

        # Generate the distance matrix for compute vina score.
        vina_dist = torch.zeros(len(current_r_xs))
        for num in range(len(current_r_xs)):
            vina_dist[num] = self.dist[rec_atom_indices[num], lig_atom_indices[num]]

        vina = VinaScoreCore(vina_dist, vina_rec_xs, vina_lig_xs)
        vina_inter_term = vina.process()
        self.vina_inter_energy = vina_inter_term / (1 + 0.05846 * (self.ligand.active_torsion + 0.5 * self.ligand.inactive_torsion))

        # inter clash
        self.inter_distance(vina_rec_xs, vina_lig_xs, vina_dist)

        return self

    def process(self):
        self.cal_RMSD()
        self.cal_vina()
        self.cal_intra_repulsion()

        rmsd_vina = self.weight_1 * self.pred_rmsd + self.weight_2 * self.vina_inter_energy

        combined_score = self.alpha * rmsd_vina + ((1 - self.alpha) / 2) * (
            torch.log(self.intra_repulsive_term) + torch.log(self.inter_repulsive_term))

        return self.pred_rmsd, self.vina_inter_energy, rmsd_vina, combined_score

class VinaScoreCore(object):

    def __init__(self, dist_matrix, rec_xs_list, lig_xs_list):
        self.dist_matrix = dist_matrix
        self.rec_xs_list = rec_xs_list
        self.lig_xs_list = lig_xs_list

        with open(dir_ + "/vdw_radii_dict.json") as f:
            self.vdw_radii_dict = json.load(f)

    def is_hydrophobic(self, atom_xs):
        return atom_xs in ["C_H", "F_H", "Cl_H", "Br_H", "I_H"]

    def is_hbdonor(self, atom_type):
        return atom_type in ["N_D", "N_DA", "O_DA", "Met_D"]

    def is_hbacceptor(self, atom_type):
        return atom_type in ["N_A", "N_DA", "O_A", "O_DA"]

    def is_hbond(self, atom_1, atom_2):
        return (
                (self.is_hbdonor(atom_1) and self.is_hbacceptor(atom_2)) or
                (self.is_hbdonor(atom_2) and self.is_hbacceptor(atom_1))
        )

    def score_function(self):
        r_xs_vdw = torch.from_numpy(np.array([self.vdw_radii_dict[x] for x in self.rec_xs_list]))
        l_xs_vdw = torch.from_numpy(np.array([self.vdw_radii_dict[x] for x in self.lig_xs_list]))

        d_ij = self.dist_matrix - (r_xs_vdw + l_xs_vdw)

        Gauss_1 = torch.sum(torch.exp(- torch.pow(d_ij / 0.5, 2)))
        Gauss_2 = torch.sum(torch.exp(- torch.pow((d_ij - 3) / 2, 2)))

        # Repulsion
        Repulsion = torch.sum(torch.pow(((d_ij < 0) * d_ij), 2))

        # Hydrophobic
        r_hydro = torch.from_numpy(np.array(list(map(self.is_hydrophobic, self.rec_xs_list))) * 1.)
        l_hydro = torch.from_numpy(np.array(list(map(self.is_hydrophobic, self.lig_xs_list))) * 1.)

        Hydro_1 = torch.sum(r_hydro * l_hydro * (d_ij <= 0.5) * 1.)

        Hydro_2_condition = r_hydro * l_hydro * (d_ij > 0.5) * (d_ij < 1.5) * 1.
        Hydro_2 = 1.5 * torch.sum(Hydro_2_condition) - torch.sum(Hydro_2_condition * d_ij)

        Hydrophobic = Hydro_1 + Hydro_2

        # HBonding
        is_hbond_tensor = torch.from_numpy(np.array(list(map(self.is_hbond, self.rec_xs_list, self.lig_xs_list))) * 1.)

        hbond_1 = is_hbond_tensor * (d_ij <= -0.7) * 1.
        hbond_2 = is_hbond_tensor * (d_ij < 0) * (d_ij > -0.7) * 1.0 * (- d_ij) / 0.7
        HBonding = torch.sum(hbond_1) + torch.sum(hbond_2)

        inter_energy = - 0.035579 * Gauss_1 - 0.005156 * Gauss_2 + 0.840245 * Repulsion - 0.035069 * Hydrophobic - 0.587439 * HBonding

        return inter_energy

    def process(self):

        final_inter_score = self.score_function()

        return final_inter_score


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
