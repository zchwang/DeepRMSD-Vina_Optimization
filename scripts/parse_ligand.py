import numpy as np
import torch
from torch import sin, cos
import itertools
import json
import os


_current_dpath = os.path.dirname(os.path.abspath(__file__))

class Ligand():
    def __init__(self, ligand_pdbqt_fpath: str = None):
        self.ligand_fpath = ligand_pdbqt_fpath

        self.heavy_atoms_previous_series = []
        self.heavy_atoms_current_index = []
        self.lig_heavy_atoms_element = [] # Elements types of heavy atoms.
        self.lig_heavy_atoms_xs_types = [] # X-Score atom types of heavy atoms.
        self.lig_heavy_atoms_ad4_types = [] # Atom types defined by AutoDock4 of heavy atoms.

        self.lig_all_atoms_ad4_types = []
        self.lig_all_atoms_xs_types = []

        self.root_heavy_atom_index = [] # The indices of heavy atoms in the root frame (the first substructure).
        self.frame_heavy_atoms_index_list = [] # The indices of heavy atoms in other substructures.
        self.frame_all_atoms_index_list = []

        self.origin_heavy_atoms_lines = []

        self.series_to_index_dict = {}
        self.torsion_bond_series = []
        self.torsion_bond_index = [] # The indices of atoms at both ends of a rotatable bond.
        self.torsion_bond_index_matrix = torch.tensor([])  # heavy_atoms

        self.init_lig_heavy_atoms_xyz = torch.tensor([])
        self.lig_all_atoms_xyz = torch.tensor([])
        self.lig_all_atoms_indices = []

        self.root_switch = False
        self.branch_switch = False
        self.number_of_H = 0
        self.number_of_branch = 0
        self.number_of_frames = 0
        self.number_of_heavy_atoms = 0

        self.ligand_center = torch.tensor([])
        self.inactive_torsion = 0

        self.lig_carbon_is_hydrophobic_dict = {}
        self.lig_atom_is_hbdonor_dict = {}

        self.updated_lig_heavy_atoms_xs_types = []
        self.frame_heavy_atoms_matrix = torch.tensor([])

        with open(os.path.join(_current_dpath, "atomtype_mapping.json")) as f:
            self.atomtype_mapping = json.load(f)

        with open(os.path.join(_current_dpath, "covalent_radii_dict.json")) as f:
            self.covalent_radii_dict = json.load(f)

        with open(os.path.join(_current_dpath, "vdw_radii_dict.json")) as f:
            self.vdw_radii_dict = json.load(f)

    def parse_ligand(self):
        # parse the ligand
        self._read_pdbqt()
        self._parse_frame()
        self.update_heavy_atoms_xs_types()
        self.update_ligand_bonded_information()
        self.generate_frame_heavy_atoms_matrix()
        self.cal_active_torsion()

        self.init_conformation_tentor()

        return self

    def _read_pdbqt(self):
        with open(self.ligand_fpath) as f:
            self.lines = [x[:-1] for x in f.readlines()]

        return self

    def _number_of_heavy_atoms_of_frame(self, ad4_types_list):
        number = sum([1 if not x in ["H", "HD"] else 0 for x in ad4_types_list])
        return number

    def _parse_frame(self):
        init_lig_heavy_atoms_xyz = []
        lig_all_atoms_xyz = []

        for line in self.lines:

            # parse root frame
            if line[:4] == "ROOT":
                self.root_switch = True
                continue

            if line[:7] == "ENDROOT":
                self.root_switch = False
                continue

            if self.root_switch == True:
                atom_num = int(line.split()[1])

                atom_ad4_type = line[77:79].strip()
                atom_xs_type = self.atomtype_mapping[atom_ad4_type]

                self.lig_all_atoms_ad4_types.append(atom_ad4_type)
                self.lig_all_atoms_xs_types.append(atom_xs_type)

                # xyz of each heavy atom
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())

                atom_xyz = np.c_[x, y, z][0]
                lig_all_atoms_xyz.append(atom_xyz)

                if atom_xs_type != "dummy":
                    self.heavy_atoms_previous_series.append(atom_num)

                    index = atom_num - (self.number_of_H + 1)
                    self.heavy_atoms_current_index.append(index)
                    self.root_heavy_atom_index.append(index)

                    init_lig_heavy_atoms_xyz.append(atom_xyz)
                    self.lig_heavy_atoms_element.append(atom_xs_type.split('_')[0])
                    self.lig_heavy_atoms_ad4_types.append(atom_ad4_type)
                    self.lig_heavy_atoms_xs_types.append(atom_xs_type)
                    self.origin_heavy_atoms_lines.append(line)
                else:
                    self.number_of_H += 1

            # parse other frame
            if line[:9] == "ENDBRANCH":
                self.branch_switch = False
                continue

            if line[:6] == "BRANCH":
                self.number_of_branch += 1

                parent_id = int(line.split()[1])
                son_id = int(line.split()[2])

                each_torsion_bond_series = [parent_id, son_id]
                self.torsion_bond_series.append(each_torsion_bond_series)

                if self.number_of_branch != 1:
                    self.frame_heavy_atoms_index_list.append(each_frame_heavy_atoms_index)
                    self.frame_all_atoms_index_list.append(each_frame_all_atoms_index)

                    each_frame_all_atoms_ad4_types = [self.lig_all_atoms_ad4_types[x] for x in
                                                      each_frame_all_atoms_index]

                    number_of_heavy_atoms_of_frame = self._number_of_heavy_atoms_of_frame(
                        each_frame_all_atoms_ad4_types)

                    if parent_id != each_frame_all_atoms_index[0] + 1 and number_of_heavy_atoms_of_frame == 1:
                        self.inactive_torsion += 1

                self.branch_switch = True
                each_frame_all_atoms_index = []
                each_frame_heavy_atoms_index = []

            if self.branch_switch == True:
                atom_num = int(line.split()[1])

                if not line[:4] == "ATOM" and not line[:6] == "HETATM":
                    continue

                atom_ad4_type = line[77:79].strip()
                atom_xs_type = self.atomtype_mapping[atom_ad4_type]

                self.lig_all_atoms_ad4_types.append(atom_ad4_type)
                self.lig_all_atoms_xs_types.append(atom_xs_type)

                # xyz of each atom
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())

                atom_xyz = np.c_[x, y, z][0]
                lig_all_atoms_xyz.append(atom_xyz)

                each_frame_all_atoms_index.append(atom_num - 1)

                if atom_xs_type != "dummy":
                    self.heavy_atoms_previous_series.append(atom_num)

                    index = atom_num - (self.number_of_H + 1)
                    self.heavy_atoms_current_index.append(index)
                    each_frame_heavy_atoms_index.append(index)

                    init_lig_heavy_atoms_xyz.append(atom_xyz)
                    self.lig_heavy_atoms_element.append(atom_xs_type.split('_')[0])
                    self.lig_heavy_atoms_ad4_types.append(atom_ad4_type)
                    self.lig_heavy_atoms_xs_types.append(atom_xs_type)
                    self.origin_heavy_atoms_lines.append(line)

                else:
                    self.number_of_H += 1

            if line[:7] == "TORSDOF":
                try:
                    self.frame_heavy_atoms_index_list.append(each_frame_heavy_atoms_index)
                    self.frame_all_atoms_index_list.append(each_frame_all_atoms_index)

                    each_frame_all_atoms_ad4_types = [self.lig_all_atoms_ad4_types[x] for x in
                                                      each_frame_all_atoms_index]

                    number_of_heavy_atoms_of_frame = self._number_of_heavy_atoms_of_frame(
                        each_frame_all_atoms_ad4_types)
                    if number_of_heavy_atoms_of_frame == 1:
                        self.inactive_torsion += 1

                except UnboundLocalError:
                    print('no other frame existed in this ligand ...')

        self.init_lig_heavy_atoms_xyz = torch.from_numpy(np.array(init_lig_heavy_atoms_xyz)).to(torch.float32)
        self.lig_all_atoms_xyz = torch.from_numpy(np.array(lig_all_atoms_xyz)).to(torch.float32)

        self.ligand_center = torch.mean(self.init_lig_heavy_atoms_xyz, dim=0)
        self.lig_all_atoms_indices = [x for x in range(len(self.lig_all_atoms_xyz))]

        return self

    def update_ligand_bonded_information(self):

        self.number_of_frames = len(self.frame_heavy_atoms_index_list)
        self.number_of_heavy_atoms = len(self.init_lig_heavy_atoms_xyz)

        self.series_to_index_dict = dict(zip(self.heavy_atoms_previous_series, self.heavy_atoms_current_index))
        self.torsion_bond_index_matrix = torch.zeros(self.number_of_heavy_atoms, self.number_of_heavy_atoms)

        for i in self.torsion_bond_series:
            Y = self.series_to_index_dict[i[0]]
            X = self.series_to_index_dict[i[1]]

            self.torsion_bond_index.append([Y, X])
            self.torsion_bond_index_matrix[Y, X] = 1
            self.torsion_bond_index_matrix[X, Y] = 1

        return self

    def _lig_carbon_is_hydrophobic(self, carbon_index, candidate_neighbors_indices):

        the_lig_carbon_is_hydrophobic = True

        if carbon_index in self.lig_carbon_is_hydrophobic_dict.keys():
            the_lig_carbon_is_hydrophobic = self.lig_carbon_is_hydrophobic_dict[carbon_index]
        else:
            for candi_neighb_index in candidate_neighbors_indices:
                if carbon_index == candi_neighb_index:
                    continue
                else:
                    candi_d = torch.sqrt(torch.sum(torch.square(
                        self.lig_all_atoms_xyz[carbon_index] - self.lig_all_atoms_xyz[candi_neighb_index])))
                    if candi_d <= self.covalent_radii_dict[self.lig_all_atoms_ad4_types[carbon_index]] + \
                            self.covalent_radii_dict[
                                self.lig_all_atoms_ad4_types[candi_neighb_index]]:

                        if not self.lig_all_atoms_ad4_types[candi_neighb_index] in ["H", "HD", "C", "A"]:
                            the_lig_carbon_is_hydrophobic = False
                            break

        self.lig_carbon_is_hydrophobic_dict[carbon_index] = the_lig_carbon_is_hydrophobic

        if the_lig_carbon_is_hydrophobic == False:
            the_lig_atom_xs = "C_P"
        else:
            the_lig_atom_xs = "C_H"

        return the_lig_atom_xs

    def _lig_atom_is_hbdonor(self, lig_atom_index, candidate_neighbors_indices):
        the_lig_atom_is_hbdonor = False

        if lig_atom_index in self.lig_atom_is_hbdonor_dict.keys():
            the_lig_atom_is_hbdonor = self.lig_atom_is_hbdonor_dict[lig_atom_index]
        else:
            for candi_neighb_index in candidate_neighbors_indices:
                if lig_atom_index == candi_neighb_index:
                    continue
                else:
                    if self.lig_all_atoms_ad4_types[candi_neighb_index] == "HD":
                        candi_d = torch.sqrt(torch.sum(torch.square(
                            self.lig_all_atoms_xyz[lig_atom_index] - self.lig_all_atoms_xyz[candi_neighb_index])))
                        if candi_d <= self.covalent_radii_dict[self.lig_all_atoms_ad4_types[lig_atom_index]] + \
                                self.covalent_radii_dict[
                                    self.lig_all_atoms_ad4_types[candi_neighb_index]]:
                            the_lig_atom_is_hbdonor = True

        self.lig_atom_is_hbdonor_dict[lig_atom_index] = the_lig_atom_is_hbdonor

        atom_xs = self.lig_all_atoms_xs_types[lig_atom_index]

        if the_lig_atom_is_hbdonor == True:
            if atom_xs == "N_P":
                atom_xs = "N_D"
            elif atom_xs == "N_A":
                atom_xs = "N_DA"
            elif atom_xs == "O_A":
                atom_xs = "O_DA"
            else:
                print("atom xs Error ...")

        return atom_xs

    def update_heavy_atoms_xs_types(self):

        for atom_index, xs in enumerate(self.lig_all_atoms_xs_types):

            if xs == "dummy":
                continue

            if xs == "C_H":
                xs = self._lig_carbon_is_hydrophobic(atom_index, self.lig_all_atoms_indices)

            # if the atom bonded a polorH, the atom xs --> HBdonor
            if xs in ["N_P", "N_A", "O_A"]:  # the ad4 types are ["N", "NA", "OA"]
                xs = self._lig_atom_is_hbdonor(atom_index, self.lig_all_atoms_indices)

            self.updated_lig_heavy_atoms_xs_types.append(xs)

        return self.updated_lig_heavy_atoms_xs_types

    def generate_frame_heavy_atoms_matrix(self):
        """
        Args:
            The indices of atoms in each frame including root

        Returns:
            Matrix, the value is 1 if the two atoms in same frame, else 0.
            # shape (N, N), N is the number of heavy atoms
        """
        self.frame_heavy_atoms_matrix = torch.zeros(len(self.lig_heavy_atoms_ad4_types),
                                                    len(self.lig_heavy_atoms_ad4_types))
        root_heavy_atoms_pairs = list(itertools.product(self.root_heavy_atom_index, self.root_heavy_atom_index))
        for i in root_heavy_atoms_pairs:
            self.frame_heavy_atoms_matrix[i[0], i[1]] = 1

        for heavy_atoms_list in self.frame_heavy_atoms_index_list:
            heavy_atoms_pairs = list(itertools.product(heavy_atoms_list, heavy_atoms_list))
            for i in heavy_atoms_pairs:
                self.frame_heavy_atoms_matrix[i[0], i[1]] = 1

        return self

    def cal_active_torsion(self):
        """Calculate number of active_torsion angles

        Returns:
            [int]: Number of active torsion angles
        """
        self.active_torsion = self.number_of_frames - self.inactive_torsion

        return self.active_torsion

    def init_conformation_tentor(self, float_: float = 0.001):
        float_ = torch.tensor(float_)

        xyz = self.init_lig_heavy_atoms_xyz[0]
        number_of_frames = self.number_of_frames

        _temp_vector = torch.zeros(6 + number_of_frames)
        self.init_cnfr = torch.zeros(6 + number_of_frames)

        for i in range(0, 3):
            self.init_cnfr[i] = xyz[i] + _temp_vector[i]

        for i in range(3, 3 + number_of_frames):
            self.init_cnfr[i] = float_ + _temp_vector[i]


        return self.init_cnfr.requires_grad_()


def rotation_matrix(alpha: float, beta: float, gamma: float) -> torch.Tensor:
    alpha = alpha.clone()
    beta = beta.clone()
    gamma = gamma.clone()
    Rx = torch.zeros(9)
    Ry = torch.zeros(9)
    Rz = torch.zeros(9)

    Rx_list = [(alpha + 1) / (alpha + 1), alpha - alpha, alpha - alpha,
               alpha - alpha, torch.cos(alpha), - torch.sin(alpha),
               alpha - alpha, torch.sin(alpha), torch.cos(alpha)]

    Ry_list = [torch.cos(beta), beta - beta, - torch.sin(beta),
               beta - beta, (beta + 1) / (beta + 1), beta - beta,
               torch.sin(beta), beta - beta, torch.cos(beta)]

    Rz_list = [torch.cos(gamma), -torch.sin(gamma), gamma - gamma,
               torch.sin(gamma), torch.cos(gamma), gamma - gamma,
               gamma - gamma, gamma - gamma, (gamma + 1) / (gamma + 1)]

    for i in range(9):
        Rx[i] = Rx_list[i]
        Ry[i] = Ry_list[i]
        Rz[i] = Rz_list[i]

    Rx = Rx.reshape(3, 3)
    Ry = Ry.reshape(3, 3)
    Rz = Rz.reshape(3, 3)

    R = torch.mm(torch.mm(Rx, Ry), Rz)

    return R


def rodrigues(vector, theta):
    theta = theta.clone()

    a = vector[0]
    b = vector[1]
    c = vector[2]

    R = torch.zeros(9)

    R_list = [
        cos(theta) + torch.pow(a, 2) * (1 - cos(theta)), a * b * (1 - cos(theta)) - c * sin(theta),
        a * c * (1 - cos(theta)) + b * sin(theta),
        a * b * (1 - cos(theta)) + c * sin(theta), cos(theta) + torch.pow(b, 2) * (1 - cos(theta)),
        b * c * (1 - cos(theta)) - a * sin(theta),
        a * c * (1 - cos(theta)) - b * sin(theta), b * c * (1 - cos(theta)) + a * sin(theta),
        cos(theta) + torch.pow(c, 2) * (1 - cos(theta))
    ]

    for i in range(9):
        R[i] = R_list[i]

    # 保证torch.mm(xyz, R), 即坐标在前，旋转矩阵在后
    R = R.reshape(3, 3).T

    return R


def vector_length(vector):
    return torch.sqrt(torch.sum(torch.square(vector)))


def relative_vector_rotation(vector, R):
    vec_length = vector_length(vector)

    new_vector = torch.mm(vector.reshape(1, -1), R)[0]
    new_vec_length = vector_length(new_vector)

    new_vector = new_vector * vec_length / new_vec_length

    return new_vector


if __name__ == '__main__':
    import os, sys

    ligand = Ligand(sys.argv[1])
    ligand.parse_ligand()

