from parse_ligand import Ligand, rotation_matrix, relative_vector_rotation, rodrigues
import torch.nn.functional as F
import torch

class LigandConformation(Ligand):

    """
    Translate 3D structure of ligand from 6+k vector.
    """

    def __init__(self, ligand_fpath: str = None, cnfr: str=None):

        super(LigandConformation, self).__init__(ligand_fpath)

        # load the bonding relationships between related atoms in the ligand.
        self.parse_ligand()

        # Coordinates of the heavy atoms of the initial pose.
        self.init_heavy_atoms_coords = self.init_lig_heavy_atoms_xyz

        # Initializes the heavy atom coordinates of the current optimized structure.
        self.pose_heavy_atoms_coords = torch.zeros(self.number_of_heavy_atoms, 3)

        # A container used to record rotation and torsion matrix.
        self.all_rotation_matrix = [0] * self.number_of_heavy_atoms

        # The rotation matrix of the first substructure.
        self.root_rotation_matrix = torch.tensor([])

    def _update_root_coords(self):

        """
        Update the coordinates of heavy atoms in the first substructure.

        """

        # Root frame
        root_alpha = self.pose_cnfr[3]
        root_beta = self.pose_cnfr[4]
        root_gamma = self.pose_cnfr[5]

        self.root_rotation_matrix = rotation_matrix(root_alpha, root_beta, root_gamma)

        # Update the coordinates of the first atom.
        root_first_atom_coord = self.pose_cnfr[:3]
        root_first_atom_coord = self.ligand_center + relative_vector_rotation(
            (root_first_atom_coord - self.ligand_center), self.root_rotation_matrix)

        self.pose_heavy_atoms_coords[0] = root_first_atom_coord * 1.0
        self.all_rotation_matrix[0] = self.root_rotation_matrix

        # Update the coordinates of other heavy atoms in the first substructure.
        if len(self.root_heavy_atom_index) != 1:
            for i in self.root_heavy_atom_index[1:]:
                relative_coord = self.init_heavy_atoms_coords[i] - self.init_heavy_atoms_coords[0]

                new_relative_vector = relative_vector_rotation(relative_coord, self.root_rotation_matrix)
                new_coord = root_first_atom_coord + new_relative_vector

                self.pose_heavy_atoms_coords[i] = new_coord * 1.0

                self.all_rotation_matrix[i] = self.root_rotation_matrix

        return self

    def _update_frame_coords(self, frame_id: int):

        """
        Update the coordinates of heavy atoms in other substructures.
        """

        # The atomic number at both ends of the rotatable bond.
        f_atoms_index = self.frame_heavy_atoms_index_list[frame_id - 1]

        rotorX_index = self.torsion_bond_index[frame_id - 1][0]
        rotorY_index = self.torsion_bond_index[frame_id - 1][1]

        # update the first atom in this frame
        rotorX_to_rotorY_vector = self.init_heavy_atoms_coords[rotorY_index][:3] - self.init_heavy_atoms_coords[
                                                                                       rotorX_index][:3]

        new_rotorY_coord = self.pose_heavy_atoms_coords[rotorX_index] + relative_vector_rotation(rotorX_to_rotorY_vector,
                                                                                    self.all_rotation_matrix[
                                                                                        rotorX_index])
        self.all_rotation_matrix[rotorY_index] = self.all_rotation_matrix[rotorX_index]

        self.pose_heavy_atoms_coords[rotorY_index] = new_rotorY_coord

        # update rotation
        new_rotorX_to_rotorY_vector = self.pose_heavy_atoms_coords[rotorY_index] - self.pose_heavy_atoms_coords[rotorX_index]
        tortion_axis = F.normalize(new_rotorX_to_rotorY_vector, p=2, dim=0)

        tortion_R = rodrigues(tortion_axis, self.pose_cnfr[6 + frame_id - 2])
        current_rotation_matrix = torch.mm(self.all_rotation_matrix[rotorY_index], tortion_R)

        # update other atoms in this frame
        if len(f_atoms_index) != 1:
            for i in f_atoms_index[1:]:
                relative_coord = self.init_heavy_atoms_coords[i][:3] - self.init_heavy_atoms_coords[rotorY_index][:3]
                new_coord = new_rotorY_coord + relative_vector_rotation(relative_coord, current_rotation_matrix)

                self.pose_heavy_atoms_coords[i] = new_coord * 1.0
                self.all_rotation_matrix[i] = current_rotation_matrix

        return self

    def cnfr2xyz(self, cnfr: torch.Tensor = None) -> torch.Tensor:

        """
        Args:
            cnfr: The 6+ K vector to be decoded.

        Returns:
            pose_heavy_atoms_coords: The coordinates of heavy atoms for the ligand decoded from this vector.
        """

        self.pose_cnfr = cnfr
        self._update_root_coords()

        for i in range(1, 1 + self.number_of_frames):
            self._update_frame_coords(i)

        return self.pose_heavy_atoms_coords



