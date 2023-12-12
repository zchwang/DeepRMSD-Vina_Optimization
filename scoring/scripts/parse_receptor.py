import os
import numpy as np
import torch
import json

_current_dpath = os.path.dirname(os.path.abspath(__file__))

class Receptor(object):
    """The receptor class.
    Args:
        object ([type]): [description]
    """

    def __init__(self, receptor_fpath: str = None):

        """
        The receptor class.
        
        Args:
            receptor_fpath (str, optional): Input receptor file path. Defaults to None.
        """
        
        self.receptor_fpath = receptor_fpath  # the pdbqt file of protein

        self.rec_index_to_series_dict = {}
        self.rec_atom_is_hbdonor_dict = {}
        self.rec_carbon_is_hydrophobic_dict = {}  # keys are indices which only contains the heavy atoms

        self.rec_all_atoms_ad4_types = []  # Atom types defined by AutoDock4 of all atoms.
        self.rec_all_atoms_xs_types = []  # X-Score atom types of all atoms (including H atom).
        self.rec_heavy_atoms_xs_types = []  # X-Score atom types of heavy atoms.

        self.residues = []
        self.residues_pool = []

        self.residues_heavy_atoms_pairs = []  # Atom types defined by DeepRMSD
        self.residues_heavy_atoms_indices = [] # Heavy atoms indices of each residue
        self.heavy_atoms_residues_indices = []  # The residue number where the heavy atom is located.
        self.residues_all_atoms_indices = []

        self.receptor_original_lines = []

        self.rec_all_atoms_xyz = torch.tensor([])  # Coordinates of all atoms (including H atom) in the receptor.
        self.rec_heavy_atoms_xyz = torch.tensor([])  # Coordinates of heavy atoms in the receptor.

        # side chain
        self.rec_heavy_atoms_pdb_types = []

        with open(os.path.join(_current_dpath, "atomtype_mapping.json")) as f:
            self.atomtype_mapping = json.load(f)

        with open(os.path.join(_current_dpath, "covalent_radii_dict.json")) as f:
            self.covalent_radii_dict = json.load(f)

    def _obtain_xyz(self, line: str = None) -> tuple:
        """Obtain XYZ coordinates from a pdb line.
        Args:
            line (str, optional): PDB line. Defaults to None.
        Returns:
            tuple: x, y, z
        """
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())

        return x, y, z

    def parse_receptor(self):
        """Parse the receptor pdbqt file.
        Returns:
            self: [description]
        """

        self._read_pdbqt()

        return self

    def _read_pdbqt(self):
        with open(self.receptor_fpath) as f:
            self.rec_lines = [x for x in f.readlines() if
                              (len(x) > 4 and x[:4] == "ATOM")]

        rec_heavy_atoms_xyz = []
        rec_all_atoms_xyz = []
        temp_indices = []  # the indices of all atoms in each residue
        temp_heavy_atoms_indices = []  # the indices of heavy atoms in each residue
        temp_heavy_atoms_pdb_types = []
        heavy_atom_num = -1

        for num, line in enumerate(self.rec_lines):
            atom_ad4_type = line[77:79].strip()
            atom_xs_type = self.atomtype_mapping[atom_ad4_type]
            atom_ele = atom_xs_type.split('_')[0]

            if atom_xs_type != "dummy":
                heavy_atom_num += 1

            self.rec_all_atoms_ad4_types.append(atom_ad4_type)
            self.rec_all_atoms_xs_types.append(atom_xs_type)

            x, y, z = self._obtain_xyz(line)

            atom_xyz = np.c_[x, y, z][0]
            rec_all_atoms_xyz.append(atom_xyz)

            pdb_type = line[12:16].strip()
            res_name = line[17:20]
            resid_symbol = line[17:27]

            if num == 0:
                self.residues_pool.append(resid_symbol)
                self.residues.append(res_name)

            if self.residues_pool[-1] == resid_symbol:
                temp_indices.append(num)

                if atom_xs_type != "dummy":
                    self.rec_index_to_series_dict[heavy_atom_num] = num
                    self.heavy_atoms_residues_indices.append(len(self.residues) - 1)

                    self.rec_heavy_atoms_xs_types.append(atom_xs_type)
                    rec_heavy_atoms_xyz.append(atom_xyz)
                    temp_heavy_atoms_indices.append(heavy_atom_num)
                    self.residues_heavy_atoms_pairs.append(res_name + '-' + atom_ele)
                    self.rec_heavy_atoms_pdb_types.append(pdb_type)

                    self.receptor_original_lines.append(line)

            else:
                self.residues_all_atoms_indices.append(temp_indices)
                self.residues_heavy_atoms_indices.append(temp_heavy_atoms_indices)

                self.residues_pool.append(resid_symbol)
                self.residues.append(res_name)

                temp_indices = [num]
                if atom_xs_type != "dummy":
                    self.rec_index_to_series_dict[heavy_atom_num] = num
                    self.heavy_atoms_residues_indices.append(len(self.residues) - 1)

                    self.rec_heavy_atoms_xs_types.append(atom_xs_type)
                    rec_heavy_atoms_xyz.append(atom_xyz)
                    temp_heavy_atoms_indices = [heavy_atom_num]
                    self.residues_heavy_atoms_pairs.append(res_name + '-' + atom_ele)
                    self.rec_heavy_atoms_pdb_types.append(pdb_type)

                    self.receptor_original_lines.append(line)

                else:
                    pass

            if num == len(self.rec_lines) - 1:
                self.residues_all_atoms_indices.append(temp_indices)
                self.residues_heavy_atoms_indices.append(temp_heavy_atoms_indices)


        self.rec_all_atoms_xyz = torch.from_numpy(np.array(rec_all_atoms_xyz)).to(torch.float32)
        self.rec_heavy_atoms_xyz = torch.from_numpy(np.array(rec_heavy_atoms_xyz)).to(torch.float32)

        return self

    def _rec_carbon_is_hydrophobic(self, carbon_index: int, candidate_neighbors_indices: list):
        the_rec_carbon_is_hydrophobic = True
        if carbon_index in self.rec_carbon_is_hydrophobic_dict.keys():
            the_rec_carbon_is_hydrophobic = self.rec_carbon_is_hydrophobic_dict[carbon_index]
        else:
            for candi_neighb_index in candidate_neighbors_indices:
                if carbon_index == candi_neighb_index:
                    continue
                else:
                    candi_d = torch.sqrt(torch.sum(torch.square(
                        self.rec_all_atoms_xyz[carbon_index] - self.rec_all_atoms_xyz[candi_neighb_index])))
                    if candi_d <= self.covalent_radii_dict[self.rec_all_atoms_ad4_types[carbon_index]] + \
                            self.covalent_radii_dict[
                                self.rec_all_atoms_ad4_types[candi_neighb_index]]:

                        if not self.rec_all_atoms_ad4_types[candi_neighb_index] in ["H", "HD", "C", "A"]:
                            the_rec_carbon_is_hydrophobic = False
                            break

        self.rec_carbon_is_hydrophobic_dict[carbon_index] = the_rec_carbon_is_hydrophobic

        if the_rec_carbon_is_hydrophobic == False:
            atom_xs = "C_P"
        else:
            atom_xs = "C_H"

        return atom_xs

    def _rec_atom_is_hbdonor(self, rec_atom_index, candidate_neighbors_indices):
        the_rec_atom_is_hbdonor = False

        if rec_atom_index in self.rec_atom_is_hbdonor_dict.keys():
            the_rec_atom_is_hbdonor = self.rec_atom_is_hbdonor_dict[rec_atom_index]
        else:
            for candi_neighb_index in candidate_neighbors_indices:
                if rec_atom_index == candi_neighb_index:
                    continue
                else:
                    if self.rec_all_atoms_ad4_types[candi_neighb_index] == "HD":
                        candi_d = torch.sqrt(torch.sum(torch.square(
                            self.rec_all_atoms_xyz[rec_atom_index] - self.rec_all_atoms_xyz[candi_neighb_index])))
                        if candi_d <= self.covalent_radii_dict[self.rec_all_atoms_ad4_types[rec_atom_index]] + \
                                self.covalent_radii_dict[
                                    self.rec_all_atoms_ad4_types[candi_neighb_index]]:
                            the_rec_atom_is_hbdonor = True

        self.rec_atom_is_hbdonor_dict[rec_atom_index] = the_rec_atom_is_hbdonor

        atom_xs = self.rec_all_atoms_xs_types[rec_atom_index]

        if the_rec_atom_is_hbdonor == True:
            if atom_xs == "N_P":
                atom_xs = "N_D"
            elif atom_xs == "N_A":
                atom_xs = "N_DA"
            elif atom_xs == "O_A":
                atom_xs = "O_DA"
            else:
                print("atom xs Error ...")

        return atom_xs

    def update_rec_xs(self, r_xs: str, rec_atom_index: int,
                      previous_series: int, residue_index: int):

        """
        Upgrade the xs atom types of some atoms in the protein:
        1. "C_H" is kept if the carbon atom is not bonded to the heteto atoms (H, non-carbon heavy atoms),
        otherwise return "C_P".
        2. If a nitrogen or oxygen atom is bonded to a polar hydrogen, it is considered a hydrogen bond donor.
        """
        if r_xs == "C_H":
            if previous_series in self.rec_carbon_is_hydrophobic_dict.keys():
                r_xs = self.rec_carbon_is_hydrophobic_dict[previous_series]

            else:

                if residue_index == 0:
                    # the indices in all atoms system
                    candidate_neighbors_indices = self.residues_all_atoms_indices[residue_index] + \
                                                  self.residues_all_atoms_indices[residue_index + 1]
                elif residue_index == len(self.residues_heavy_atoms_indices) - 1:
                    candidate_neighbors_indices = self.residues_all_atoms_indices[residue_index] + \
                                                  self.residues_all_atoms_indices[residue_index - 1]
                else:
                    candidate_neighbors_indices = self.residues_all_atoms_indices[residue_index] + \
                                                  self.residues_all_atoms_indices[residue_index - 1] + \
                                                  self.residues_all_atoms_indices[residue_index + 1]

                r_xs = self._rec_carbon_is_hydrophobic(previous_series, candidate_neighbors_indices)
                self.rec_carbon_is_hydrophobic_dict[previous_series] = r_xs
                self.rec_heavy_atoms_xs_types[int(rec_atom_index)] = r_xs


        elif r_xs in ["N_P", "N_A", "O_A"]:

            if previous_series in self.rec_atom_is_hbdonor_dict.keys():
                r_xs = self.rec_atom_is_hbdonor_dict[previous_series]

            else:

                r_xs = self._rec_atom_is_hbdonor(previous_series, self.residues_all_atoms_indices[residue_index])

                self.rec_atom_is_hbdonor_dict[previous_series] = r_xs
                self.rec_heavy_atoms_xs_types[int(rec_atom_index)] = r_xs

        else:
            pass

        return self.rec_heavy_atoms_xs_types