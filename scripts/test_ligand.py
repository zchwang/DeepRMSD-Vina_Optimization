from parse_ligand import Ligand
from conformation_to_xyz import LigandConformation
from scoring_function import ScoringFunction
from parse_receptor import Receptor
from model import CNN

poses_dpath = "../samples/1gpn/decoys"
receptor_fpath = "../samples/1gpn/1gpn_protein_atom_noHETATM.pdbqt"
mean_std_file = "../models/r6-r1_0.3-2.0nm_train_mean_std.csv"
model_fpath = "../models/bestmodel_cpu.pth"

ligand = LigandConformation(poses_dpath)
#ligand = Ligand(poses_dpath)
receptor = Receptor(receptor_fpath)
receptor._read_pdbqt()

#ligand.parse_ligand()
init_cnfr = ligand.init_cnfr
#print(ligand.init_cnfr)

ligand.cnfr2xyz(init_cnfr)
#print(ligand.pose_heavy_atoms_coords)

score = ScoringFunction(receptor=receptor, ligand=ligand, mean_std_file=mean_std_file, model_fpath=model_fpath)
score.generate_pldist_mtrx()
#score.cal_RMSD()
pred_rmsd, vina_inter_energy, rmsd_vina, combined_score = score.process()
print(ligand.poses_file_names)
print(ligand.pose_heavy_atoms_coords)

print("pred rmsd:", score.pred_rmsd)
print("vina:", vina_inter_energy)
print("rmsd_vina:", rmsd_vina)
print("combined score:", combined_score)
