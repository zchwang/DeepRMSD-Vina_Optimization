import os, sys
from conformation_to_xyz import LigandConformation
from parse_receptor import Receptor
from utils import *
from scoring_function import ScoringFunction
import torch

class Optimize(object):
    def __init__(self,
                 output_dpath: str = None,
                 ligand: LigandConformation = None,
                 receptor: Receptor = None,
                 model_fpath: str = None,
                 mean_std_file: str = None,
                 epochs: int = 150,
                 patience: int = 30,
                 delta: float = 0.001,
                 torsion_param: float = 2.0,
                 ):

        self.ligand = ligand
        self.receptor = receptor
        self.model_fpath = model_fpath
        self.mean_std_file = mean_std_file
        self.output_dpath = output_dpath
        self.epochs = epochs  # Maximum number of optimizations
        self.patience = patience
        self.delta = delta
        self.torsion_param = torsion_param

        self.scores_data = []
        self.optimal_scores = torch.tensor([]) # shape [N, ] the number of poses.

        # output parameters
        self.output_dpath = output_dpath
        os.makedirs(output_dpath, exist_ok=True)

    def cal_energy(self):

        score = ScoringFunction(receptor = self.receptor, ligand = self.ligand,
                                mean_std_file = self.mean_std_file, model_fpath = self.model_fpath)
        pred_rmsd, vina, rmsd_vina, combined_score = score.process()

        return pred_rmsd, vina, rmsd_vina, combined_score
    
    def run(self):
        self.cnfr = self.ligand.init_cnfr  # Initialize 6+k vector:  xyz -> cnfr

        for step in range(self.epochs):
            print("step:", step)

            # update xyz for ligand: cnfr -> xyz
            self.ligand.cnfr2xyz(self.cnfr)

            # scoring
            pred_rmsd, vina_score, rmsd_vina, combined_score = self.cal_energy()

            # save trajectory
            output_ligand_traj(self.output_dpath, self.ligand)

            # save the predicted and real RMSD
            _score_data = np.c_[
                vina_score.detach().numpy(), pred_rmsd.detach().numpy(), rmsd_vina.detach().numpy()]
            self.scores_data.append(_score_data)

            save_data(self.scores_data, self.output_dpath, self.ligand)

            # Save optimal score and conformation for each pose
            _rmsd_vina = torch.from_numpy(rmsd_vina.reshape(-1, ).detach().numpy())
            if step == 69:
                self.optimal_scores = _rmsd_vina
                _condition = torch.ones(self.optimal_scores.shape)
              
                save_results(_condition, self.output_dpath, self.ligand, self.scores_data)
                save_final_lig_cnfr(_condition, self.output_dpath, self.ligand)
                
            elif step > 69:
                _condition = (_rmsd_vina <= self.optimal_scores - self.delta) * 1.
                self.optimal_scores = _condition * _rmsd_vina + (_condition == 0.) * self.optimal_scores

                save_results(_condition, self.output_dpath, self.ligand, self.scores_data)
                save_final_lig_cnfr(_condition, self.output_dpath, self.ligand)
            else:
                pass

            # backward
            #print("combined_score:", combined_score)
            combined_score.backward(torch.ones_like(combined_score), retain_graph=True)

            cnfr_grad = self.cnfr.grad
            cnfr_grad = 2 * torch.sigmoid(cnfr_grad) - 1  # gradient clipping

            lr = local_set_lr(step, self.torsion_param, self.ligand.number_of_frames)  # set learning rate
            self.cnfr = self.cnfr - cnfr_grad * lr

            cnfr_grad.data.zero_()
            self.cnfr = self.cnfr.detach().numpy()

            self.cnfr = torch.from_numpy(self.cnfr).requires_grad_()

def run_optimization(rec_fpath: str,
        poses_dpath: str,
        model_fpath: str,
        mean_std_file: str,
        output_path: str):

    # parse receptor
    receptor = Receptor(rec_fpath)
    receptor.parse_receptor()
    print("Receptor parsed")

    # parse ligand
    ligand = LigandConformation(poses_dpath)
    print("Ligand parsed")

    optimizer = Optimize(output_dpath=output_path,
                        receptor=receptor,
                        ligand=ligand,
                        model_fpath=model_fpath,
                        mean_std_file=mean_std_file)
    optimizer.run()
