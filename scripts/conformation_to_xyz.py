from turtle import pos
from utils import rotation_matrix, relative_vector_rotation, relative_vector_center_rotation, rodrigues
from parse_ligand import Ligand
import torch.nn.functional as F
import torch

class LigandConformation(Ligand):
    """
    Translate 3D structure of ligand from 6+k vector.
    """

    def __init__(self, poses_dpath: str = None, cnfr: str = None):

        super(LigandConformation, self).__init__(poses_dpath)

        # load the bonding relationships between related atoms in the ligand.
        self.parse_ligand()

        # The indices of heavy atoms in root frame
        self.root_heavy_atom_index = self.root_heavy_atom_index

        # Coordinates of the heavy atoms of the initial pose.
        self.init_heavy_atoms_coords = self.init_lig_heavy_atoms_xyz

        # Initializes the heavy atom coordinates of the current optimized structure.
        self.pose_heavy_atoms_coords = [0] * self.number_of_heavy_atoms

        # A container used to record torsion matrix.
        self.all_torsion_matrix = [0] * self.number_of_heavy_atoms

        # The rotation matrix of the first substructure.
        self.root_rotation_matrix = torch.tensor([])

        self.number_of_cnfr_tensor = 0

    def _update_root_coords(self):

        """
        Update the coordinates of heavy atoms in the first substructure.

        """

        root_alpha = self.cnfr_tensor[:, 3]  # shape [-1, ]
        root_beta = self.cnfr_tensor[:, 4]
        root_gamma = self.cnfr_tensor[:, 5]

        self.root_rotation_matrix = rotation_matrix(root_alpha, root_beta, root_gamma)  # shape [-1, 3, 3]

        # Update the coordinates of the first atom.
        root_first_atom_coord = self.cnfr_tensor[:, :3]  # shape [-1, 3]
        root_first_atom_coord = self.ligand_center + relative_vector_rotation(
            (root_first_atom_coord - self.ligand_center), self.root_rotation_matrix)  # shape [-1, 3]

        self.pose_heavy_atoms_coords[0] = root_first_atom_coord

        # Update the coordinates of other heavy atoms in the first substructure.
        if len(self.root_heavy_atom_index) != 1:
            for i in self.root_heavy_atom_index[1:]:
                relative_vector = self.init_heavy_atoms_coords[:, i] - self.init_heavy_atoms_coords[:, 0] # shape [-1, 3]
                
                new_relative_vector = relative_vector_center_rotation(relative_vector, self.ligand_center,
                                                                    self.root_rotation_matrix)  # shape [-1, 3]
                new_coord = root_first_atom_coord + new_relative_vector
                self.pose_heavy_atoms_coords[i] = new_coord  # shape [-1, 3]

        return self

    def _update_frame_coords(self, frame_id: int):

        """
        Update the coordinates of heavy atoms in other substructures.
        frame_id: from 1 to k.

        """

        # The atomic number at both ends of the rotatable bond.
        f_atoms_index = self.frame_heavy_atoms_index_list[frame_id - 1]

        rotorX_index = self.torsion_bond_index[frame_id - 1][0]
        rotorY_index = self.torsion_bond_index[frame_id - 1][1]

        # update the first atom in this frame
        rotorX_to_rotorY_vector = self.init_heavy_atoms_coords[:, rotorY_index] - \
            self.init_heavy_atoms_coords[:, rotorX_index]  # shape [-1, 3]
        
        # rotation
        new_relative_vector = relative_vector_center_rotation(rotorX_to_rotorY_vector, self.ligand_center,
                                                                self.root_rotation_matrix)  # shape [-1, 3]    

        # torsion
        if rotorX_index in self.root_heavy_atom_index:
            pass
        else:
            new_relative_vector = relative_vector_rotation(new_relative_vector, self.all_torsion_matrix[rotorX_index])

        new_rotorY_coord = self.pose_heavy_atoms_coords[rotorX_index] + new_relative_vector  # shape [-1, 3]

        self.pose_heavy_atoms_coords[rotorY_index] = new_rotorY_coord

        # update torsion in all_torsion_matrix
        new_rotorX_to_rotorY_vector = self.pose_heavy_atoms_coords[rotorY_index] - self.pose_heavy_atoms_coords[
            rotorX_index]  # shape [-1, 3]
        torsion_axis = F.normalize(new_rotorX_to_rotorY_vector, p=2, dim=1)  

        torsion_R = rodrigues(torsion_axis, self.cnfr_tensor[:, 6 + frame_id - 1])  # shape [-1, 3, 3]

        if rotorX_index in self.root_heavy_atom_index:
            current_torsion_R = torsion_R
        else:
            current_torsion_R = torch.matmul(self.all_torsion_matrix[rotorX_index], torsion_R)
        self.all_torsion_matrix[rotorY_index] = current_torsion_R  # shape [-1, 3, 3]

        # update other atoms in this frame
        if len(f_atoms_index) != 1:
            for i in f_atoms_index:
                if i == rotorY_index:
                    continue

                relative_vector = self.init_heavy_atoms_coords[:, i] - self.init_heavy_atoms_coords[:, rotorY_index] # shape [-1, 3]

                # rotation
                relative_vector = relative_vector_center_rotation(relative_vector, self.ligand_center,
                                                                  self.root_rotation_matrix)  # shape [-1, 3]

                # torsion
                relative_vector = relative_vector_rotation(relative_vector, current_torsion_R)  # shape [-1, 3]
                new_coord = new_rotorY_coord + relative_vector
                self.pose_heavy_atoms_coords[i] = new_coord
                self.all_torsion_matrix[i] = current_torsion_R
    
        return self

    def cnfr2xyz(self, cnfr_tensor: torch.Tensor = None) -> torch.Tensor:

        """
        Args:
            cnfr: The 6+ K vector to be decoded.

        Returns:
            pose_heavy_atoms_coords: The coordinates of heavy atoms for the ligand decoded from this vector.
            shape [N, M, 3], where N is the number of cnfr, and M is the number of atoms in this ligand.
        """

        self.number_of_cnfr_tensor = len(cnfr_tensor)
        self.pose_heavy_atoms_coords = [0] * self.number_of_heavy_atoms

        self.cnfr_tensor = cnfr_tensor
        self._update_root_coords()

        for i in range(1, 1 + self.number_of_frames):
            self._update_frame_coords(i)

        self.pose_heavy_atoms_coords = torch.cat(self.pose_heavy_atoms_coords, axis=1).reshape(len(self.cnfr_tensor),
                                                                                                -1, 3)

        return self.pose_heavy_atoms_coords