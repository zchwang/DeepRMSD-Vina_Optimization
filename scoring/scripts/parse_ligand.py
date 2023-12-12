from turtle import pos
import numpy as np
import torch
import itertools
import json
import os

_current_dpath = os.path.dirname(os.path.abspath(__file__))

class Ligand():
    def __init__(self, poses_file: str=None):
        #self.poses_dpath = poses_dir
        self.poses_file = poses_file  # this file could contain mutiple poses 

        self.heavy_atoms_previous_series = []  # The indices of heavy atoms in pdbqt file
        self.heavy_atoms_current_index = []  # The indices of heavy atoms when not including H.
        self.lig_heavy_atoms_element = []  # Elements types of heavy atoms.
        self.lig_heavy_atoms_xs_types = []  # X-Score atom types of heavy atoms.
        self.lig_heavy_atoms_ad4_types = []  # Atom types defined by AutoDock4 of heavy atoms.

        self.lig_all_atoms_ad4_types = []  # The AutoDock atom types of all atoms
        self.lig_all_atoms_xs_types = []  # The X-score atom types of all atoms

        self.root_heavy_atom_index = []  # The indices of heavy atoms in the root frame (the first substructure).
        self.frame_heavy_atoms_index_list = []  # The indices of heavy atoms in other substructures (Root is not included).
        self.frame_all_atoms_index_list = []  # The indices of all atoms in other substructures (Root is not included).

        self.origin_heavy_atoms_lines = []  # In the pdbqt file, the content of the lines where heavy atoms are located.

        self.series_to_index_dict = {}  # Dict: keys: self.heavy_atoms_previous_series; values: self.heavy_atoms_current_index
        self.torsion_bond_series = []  # The atomic index of the rotation bond is the index
        self.torsion_bond_index = []  # The indices of atoms at both ends of a rotatable bond.
        self.torsion_bond_index_matrix = torch.tensor([])  # heavy_atoms

        self.init_lig_heavy_atoms_xyz = torch.tensor([])  # The initial xyz of heavy atoms in the ligand
        self.lig_all_atoms_xyz = torch.tensor([])  # The initial xyz of all atoms in the ligand
        self.lig_all_atoms_indices = []  # The indices of all atoms (including H) in the ligand.

        self.number_of_H = 0  # The number of Hydrogen atoms.
        self.number_of_frames = 0  # The number of sub-frames (not including Root).
        self.number_of_heavy_atoms = 0  # The number of heavy atoms in the ligand.
        self.number_of_heavy_atoms_in_every_frame = []  # The number of heavy atoms in each frame (not including Root).

        self.ligand_center = torch.tensor([]) # shape [N, 3]
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

        self._get_poses_fpath()

        # parse the ligand
        self._parse_frame(self.poses_content[0])

        for num, lines in enumerate(self.poses_content[1:]):
            self._get_xyz(num+1, lines)

        self.update_heavy_atoms_xs_types()
        self.update_ligand_bonded_information()
        self.generate_frame_heavy_atoms_matrix()
        self.cal_active_torsion()

        return self

    def _get_poses_fpath(self):

        pose_basename = os.path.basename(self.poses_file).split(".")[0]

        with open(self.poses_file) as f:
            lines = [x.strip() for x in f.readlines()]

        MODEL_lines = [num for num, x in enumerate(lines) if x.startswith("MODEL")]
        ENDMODEL_lines = [num for num, x in enumerate(lines) if x.startswith("ENDMDL")]

        self.poses_file_names = []
        self.poses_content = []
        for num, (_s, _e) in enumerate(zip(MODEL_lines, ENDMODEL_lines)):
            _content = lines[_s:_e+1]
            self.poses_content.append(_content)
            self.poses_file_names.append(pose_basename + "-" + str(num+1))
        self.number_of_poses = len(self.poses_file_names)

        return self

    def _number_of_heavy_atoms_of_frame(self, ad4_types_list):
        number = sum([1 if not x in ["H", "HD"] else 0 for x in ad4_types_list])
        return number

    def _get_xyz(self, num, _lines):
       
        lines = [x.strip() for x in _lines if x[:4] == "ATOM" or x[:4] == "HETA"]

        # xyz of each heavy atom
        ha_xyz = []
        all_xyz = []
        for line in lines:
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            _xyz = np.c_[x, y, z][0]
            all_xyz.append(_xyz)
        
        for k, v in zip(self.lig_all_atoms_xs_types, all_xyz):
            if k != "dummy":
                ha_xyz.append(v)
        
        ha_xyz = torch.from_numpy(np.array(ha_xyz))
        all_xyz = torch.from_numpy(np.array(all_xyz))

        ligand_center = torch.mean(ha_xyz, dim=0)
        self.ligand_center[num] = ligand_center

        self.init_lig_heavy_atoms_xyz[num] = ha_xyz
        self.lig_all_atoms_xyz[num] = all_xyz

        return self


    def _parse_frame(self, _lines):
        lines = [x.strip() for x in _lines]

        init_lig_heavy_atoms_xyz = []
        lig_all_atoms_xyz = []

        branch_start_numbers = []
        root_start_number = 0
        root_end_number = 0
        for num, line in enumerate(lines):
            if line.startswith("ROOT"):
                root_start_number = num
            if line.startswith("ENDROOT"):
                root_end_number = num
            if line.startswith("BRANCH"):
                branch_start_numbers.append(num)

        # Root
        root_lines = lines[root_start_number + 1:root_end_number]
        for line in root_lines:
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

        # Other frames
        number_of_branch = len(branch_start_numbers)
        for num, start_num in enumerate(branch_start_numbers):

            branch_line = lines[start_num]

            parent_id = int(branch_line.split()[1])
            son_id = int(branch_line.split()[2])

            each_torsion_bond_series = [parent_id, son_id]
            self.torsion_bond_series.append(each_torsion_bond_series)

            if num == number_of_branch - 1:
                _the_branch_lines = [x.strip() for x in lines[start_num:] if
                                     x.startswith("ATOM") or x.startswith("HETATM")]

            else:
                end_num = branch_start_numbers[num + 1]
                _the_branch_lines = [x.strip() for x in lines[start_num:end_num] if
                                     x.startswith("ATOM") or x.startswith("HETATM")]

            each_frame_all_atoms_index = []
            each_frame_heavy_atoms_index = []

            for line in _the_branch_lines:
                atom_num = int(line.split()[1])

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

                number_of_heavy_atom = 0
                if atom_xs_type != "dummy":
                    number_of_heavy_atom += 1

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

            self.number_of_heavy_atoms_in_every_frame.append(number_of_heavy_atom)

            self.frame_all_atoms_index_list.append(each_frame_all_atoms_index)
            self.frame_heavy_atoms_index_list.append(each_frame_heavy_atoms_index)

        init_lig_heavy_atoms_xyz = torch.from_numpy(np.array(init_lig_heavy_atoms_xyz)).to(torch.float32)
        lig_all_atoms_xyz = torch.from_numpy(np.array(lig_all_atoms_xyz)).to(torch.float32)

        self.init_lig_heavy_atoms_xyz = torch.zeros(self.number_of_poses, len(init_lig_heavy_atoms_xyz), 3)
        self.lig_all_atoms_xyz = torch.zeros(self.number_of_poses, len(lig_all_atoms_xyz), 3)
        
        self.init_lig_heavy_atoms_xyz[0] = init_lig_heavy_atoms_xyz
        self.lig_all_atoms_xyz[0] = lig_all_atoms_xyz

        ligand_center = torch.mean(init_lig_heavy_atoms_xyz, dim=0)
        self.ligand_center = torch.zeros(self.number_of_poses, 3)
       
        self.ligand_center[0] = ligand_center

        self.lig_all_atoms_indices = [x for x in range(len(lig_all_atoms_xyz))]

        return self

    def update_ligand_bonded_information(self):

        self.number_of_frames = len(self.frame_heavy_atoms_index_list)
        self.number_of_heavy_atoms = len(self.init_lig_heavy_atoms_xyz[0])

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
                        self.lig_all_atoms_xyz[0][carbon_index] - self.lig_all_atoms_xyz[0][candi_neighb_index])))
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
                            self.lig_all_atoms_xyz[0][lig_atom_index] - self.lig_all_atoms_xyz[0][candi_neighb_index])))
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
            The indices of atoms in each frame including root.

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
        all_rotorX_indices = [bond[0] for bond in self.torsion_bond_index]
        for each_list in self.frame_heavy_atoms_index_list:
            if len(each_list) == 1 and each_list[0] not in all_rotorX_indices:
                self.inactive_torsion += 1

        self.active_torsion = self.number_of_frames - self.inactive_torsion

        return self


if __name__ == '__main__':
    import os, sys

    ligand = Ligand(sys.argv[1])
    ligand.parse_ligand()
