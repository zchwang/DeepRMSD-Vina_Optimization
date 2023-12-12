import os
import torch as th
import numpy as np
import pandas as pd
import itertools
from argparse import RawDescriptionHelpFormatter
import argparse
from spyrmsd import io, rmsd
from utils import all_defined_residues, all_rec_defined_ele, ad4_to_ele_dict, all_lig_ele, get_elementtype, \
    generate_secret, sdf_split, convert_format
#from scipy.spatial.distance import cdist

class ReceptorFile():
    def __init__(self, rec_fpath: str=None):
        with open(rec_fpath) as f:
            self.lines = [x for x in f.readlines() if x.startswith("ATOM")]

        self.rec_ha_types = []
        self.rec_ha_xyz = np.array([])

    def load_rec(self):
        ha_xyz_list = []
        for line in self.lines:
            ele = line.split()[-1]
            if ele == "H":
                continue
            res = line[17:20].strip()
            if not res in all_defined_residues:
                res = "OTH"
            if not ele in all_rec_defined_ele:
                ele = "DU"
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())

            ha_xyz_list.append(np.array([[x, y, z]]))
            self.rec_ha_types.append(res + "-" + ele)
        self.rec_ha_xyz = np.concatenate(ha_xyz_list, axis=0).astype(np.float32)

        return self
    
class LigandFile():
    def __init__(self, lig_fpath):
        self.lig_fpath = lig_fpath
        self.lig_ha_ele = []
        self.all_pose_xyz = np.array([]) # [N_pose, N_ha, 3]
        
    def parse_lig(self):
        with open(self.lig_fpath) as f:
            lines = f.readlines()

        all_pose_xyz_list = []
        pose_idx = 0
        for line in lines:
            if line.startswith("MODEL"):
                pose_idx += 1
                ha_xyz = []
                ha_ele = []
            if line.startswith("ENDMDL"):
                _pose_xyz = np.concatenate(ha_xyz, axis=0)
                all_pose_xyz_list.append(_pose_xyz.reshape(1, -1, 3))
                if pose_idx == 1:
                    self.lig_ha_ele = ha_ele.copy()

            if not line.startswith("ATOM") and not line.startswith("HETATM"):
                continue

            ad4_type = line.split()[-1]
            if ad4_type in ["H", "HD"]:
                continue
            ele = ad4_to_ele_dict[ad4_type] if ad4_type in list(ad4_to_ele_dict.keys()) else "DU"
            ele = get_elementtype(ele) 
            ha_ele.append(ele)

            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            ha_xyz.append(np.array([[x, y, z]]))
        self.all_pose_xyz = np.concatenate(all_pose_xyz_list, axis=0).astype(np.float32)

        return self

class GetFeatures():
    def __init__(self, 
                 receptor: ReceptorFile=None, 
                 ligand: LigandFile=None,
                 pre_cut: float=0.3,
                 cutoff: float=2.0):
        self.rec = receptor
        self.lig = ligand
        self.pre_cut = pre_cut
        self.cutoff = cutoff
        self.keys = self.get_defined_pairs()

    def get_defined_pairs(self):
        defined_res_pairs = ["-".join(x) for x in list(itertools.product(all_defined_residues, all_rec_defined_ele))]
        keys = []
        for r, a in list(itertools.product(defined_res_pairs, all_lig_ele)):
            keys.append(f"r6_{r}_{a}")
            keys.append(f"r1_{r}_{a}")
        return keys
    
    def get_plmtx(self):
        """
        Generate the protein-ligand distance matrix
        """
        self.number_of_poses = self.lig.all_pose_xyz.shape[0]
        rec_ha_xyz = np.repeat(self.rec.rec_ha_xyz.reshape(1, -1, 3), self.number_of_poses, axis=0)
        self.plmtx = th.cdist(th.from_numpy(rec_ha_xyz), th.from_numpy(self.lig.all_pose_xyz), p=2).numpy()
        
        #self.plmtx = cdist(rec_ha_xyz, self.lig.all_pose_xyz, metric='euclidean')

        return self
    
    def generate_features(self):
        self.get_plmtx()
        dist_nm = self.plmtx / 10
        dist_nm_1 = (dist_nm <= self.pre_cut) * self.pre_cut
        dist_nm_2 = dist_nm * (dist_nm > self.pre_cut) * (dist_nm < self.cutoff)

        features_matrix_1 = np.power(dist_nm_1 + (dist_nm_1 == 0.) * 1., -6) - (dist_nm_1 == 0.) * 1.
        features_matrix_2 = np.power(dist_nm_2 + (dist_nm_2 == 0.) * 1., -6) - (dist_nm_2 == 0.) * 1.
        features_1 = (features_matrix_1 + features_matrix_2).reshape(-1, 1)

        features_matrix_1 = np.power(dist_nm_1 + (dist_nm_1 == 0.) * 1., -1) - (dist_nm_1 == 0.) * 1.
        features_matrix_2 = np.power(dist_nm_2 + (dist_nm_2 == 0.) * 1., -1) - (dist_nm_2 == 0.) * 1.
        features_2 = (features_matrix_1 + features_matrix_2).reshape(-1, 1)     

        features = np.concatenate([features_1, features_2], axis=1).reshape(self.number_of_poses, 1, -1)
        _rec_lig_pairs = ["_".join(x) for x in list(itertools.product(self.rec.rec_ha_types, self.lig.lig_ha_ele))]
        rec_lig_pairs = []
        for i in _rec_lig_pairs:
            rec_lig_pairs.append("r6_" + i)
            rec_lig_pairs.append("r1_" + i)
        
        init_matrix = np.zeros((len(rec_lig_pairs), 1470))
        for num, c in enumerate(rec_lig_pairs):
            key_num = self.keys.index(c)
            init_matrix[num][key_num] = 1
        init_matrix = np.repeat(init_matrix.reshape(1, -1, 1470), self.number_of_poses, axis=0)
        final_features = np.matmul(features, init_matrix).reshape(-1, 1470)

        return final_features

def cal_rmsd(ref_lig_fpath, pose_fpath, temp_dir="temp_files"):
    # Convert the pdbqt files of poses to sdf format
    
    ref = io.loadmol(ref_lig_fpath)
    ref.strip()
    coords_ref = ref.coordinates
    anum_ref = ref.atomicnums
    
    string = generate_secret()
    convert_format(pose_fpath, "pdbqt", f"{temp_dir}/{string}.sdf", "sdf")
    mol_content = sdf_split(f"{temp_dir}/{string}.sdf")
   
    rmsd_list = []
    for idx, content in enumerate(mol_content):
        p_file = f"{temp_dir}/{string}-{idx+1}.sdf"
        with open(p_file, "w") as f:
            f.write(content)
        mol = io.loadmol(p_file)
        mol.strip()
        coord = mol.coordinates
        anum = mol.atomicnums
        RMSD = rmsd.hrmsd(coords_ref, coord, anum_ref, anum, center=False)
        rmsd_list.append(RMSD)
        os.remove(p_file)
    os.remove(f"{temp_dir}/{string}.sdf")

    return np.array(rmsd_list)

if __name__ == "__main__":

    print("Start Now ... ")

    d = """
        Generate DeepRMSD datasets. 
        """
    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-inp", type=str, default="input.dat",
                        help="Input. Each line specifies the path for a protein-ligand pair.")
    parser.add_argument("-pre_cut", type=float, default=0.3,
                        help="Input. Minimum distance threshold.")
    parser.add_argument("-cutoff", type=float, default=2.0,
                        help="Input. Maximum distance threshold.")
    parser.add_argument("-temp_dir", type=str, default="temp_files",
                        help="Input, optional. Path to store the temp files.")
    parser.add_argument("-out", type=str, default="features_label.pkl",
                        help="Output. Path to save dataset.")
    args = parser.parse_args()

    temp_dir = args.temp_dir
    os.makedirs(temp_dir, exist_ok=True)

    with open(args.inp) as f:
        inputs = [x.strip() for x in f.readlines() if not x.startswith("#")]
    
    final_indices = []
    final_values = []
    columns = []
    for num, inp in enumerate(inputs):
        print(num, len(inp), inp)
        target, rec_fpath, pose_fpath, ref_lig_fpath = inp.split()

        receptor = ReceptorFile(rec_fpath)
        receptor.load_rec()

        ligand = LigandFile(pose_fpath)
        ligand.parse_lig()

        feat = GetFeatures(receptor=receptor, ligand=ligand)
        features = feat.generate_features()
        real_rmsd = cal_rmsd(ref_lig_fpath, pose_fpath, temp_dir)
        feat_label = np.concatenate([features, real_rmsd.reshape(-1, 1)], axis=1)
        
        pose_basename = os.path.basename(pose_fpath).split(".")[0]
        index = [f"{pose_basename}-{i+1}" for i in range(feat_label.shape[0])]
        final_indices += index
        final_values.append(feat_label)

        if num == 0:
            columns = feat.keys
    final_values = np.concatenate(final_values, axis=0)
    final_data = pd.DataFrame(final_values, index=final_indices, columns=columns+["rmsd"])

    final_data.to_pickle(args.out)
