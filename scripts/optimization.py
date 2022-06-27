import os, sys
from conformation_to_xyz import LigandConformation
from parse_receptor import Receptor
from utils import *
from scoring_function import ScoringFunction
import torch

class Optimize(object):
    def __init__(self,
                 output_dpath: str = None,
                 native_pose_fpath: str = None,
                 ligand: LigandConformation = None,
                 receptor: Receptor = None,
                 model_fpath: str = None,
                 mean_std_file: str = None,
                 epochs: int = 200,
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
        self.native_pose = native_pose_fpath  # reference native pose
        self.patience = patience
        self.delta = delta
        self.torsion_param = torsion_param

        # variables
        self.preds_trues = []
        self.lowest_pred = [0]
        self.count_number = 0

        # output parameters
        os.makedirs(output_dpath, exist_ok=True)
        self.output_traj_fpath = os.path.join(output_dpath, "optimized_traj.pdb")
        self.output_temp_fpath = os.path.join(output_dpath, "current_pose.pdb")
        self.output_data_fpath = os.path.join(output_dpath, "opt_data.csv")
        self.output_optimized_cnfr = os.path.join(output_dpath, "final_optimized_cnfr.pdb")
        self.output_optimized_score = os.path.join(output_dpath, "final_score.csv")

    def _calculate_rmsd(self, ref: str, query: str) -> float:
        """Calculate pose rmsd, if self.native_pose is not None.

        Returns:
            real_rmsd: float, the RMSD of the newly generated pose.
        """

        try:
            real_rmsd = cal_hrmsd(ref, query)
        except:
            real_rmsd = 0.0

        return real_rmsd

    def cal_energy(self, step):

        score = ScoringFunction(step, receptor = self.receptor, ligand = self.ligand,
                                mean_std_file = self.mean_std_file, model_fpath = self.model_fpath)
        pred_rmsd, vina, rmsd_vina, combined_score = score.process()

        return pred_rmsd, vina, rmsd_vina, combined_score

    def _is_earlystopping(self, step, rmsd_vina, pred_true):
        """
                After 70 steps of optimization, earlystopping was introduced. If there is no significant improvement
            within the next 30 optimizations, the optimization stops.

        Args:
            step: number of optimizations
            rmsd_vina: rmsd_vina score

        Returns:
            _is_stopping
        """

        _is_stopping = False

        if step == 69:
            self.lowest_pred[0] = rmsd_vina
            self.count_number = 0

            save_results(self.output_optimized_score, self.preds_trues[0], pred_true)
            save_final_cnfr(self.output_optimized_cnfr, self.ligand)

        if step > 69:
            if rmsd_vina <= self.lowest_pred[0] - self.delta:
                self.lowest_pred[0] = rmsd_vina
                self.count_number = 0
                save_results(self.output_optimized_score, self.preds_trues[0], pred_true)
                save_final_cnfr(self.output_optimized_cnfr, self.ligand)
            else:
                self.count_number += 1

        if step == 69:
            self.count_number = 0

        if step > 69 and self.count_number == self.patience:
            _is_stopping = True

        return _is_stopping

    def run(self):
        self.cnfr = self.ligand.init_cnfr  # Initialize 6+k vector:  xyz -> cnfr

        for step in range(self.epochs):
            print("step:", step)

            # update xyz for ligand: cnfr -> xyz
            self.ligand.cnfr2xyz(self.cnfr)

            # scoring
            pred_rmsd, vina_score, rmsd_vina, combined_score = self.cal_energy(step)

            # save trajectory
            output_pdb(self.output_traj_fpath, self.output_temp_fpath, self.ligand)

            # calculate real_rmsd
            real_rmsd = self._calculate_rmsd(self.native_pose, self.output_temp_fpath)

            # save the predicted and real RMSD
            pred_true = np.r_[
                vina_score.detach().numpy(), pred_rmsd.detach().numpy().ravel(), rmsd_vina.detach().numpy().ravel(), real_rmsd]
            self.preds_trues.append(pred_true)

            print("vina score: %s" % str(vina_score.detach().numpy()),
                  "pred rmsd: %s" % str(pred_rmsd.detach().numpy().ravel()),
                  "rmsd_vina: %s" % str(rmsd_vina.detach().numpy().ravel()),
                  "real rmsd: %s" % str(real_rmsd))

            save_data(self.preds_trues, self.output_data_fpath)

            # EarlyStopping
            _is_stopping = self._is_earlystopping(step, rmsd_vina, pred_true)
            if _is_stopping == True:
                break

            # backward
            combined_score.backward(retain_graph=True)

            cnfr_grad = self.cnfr.grad
            cnfr_grad = 2 * torch.sigmoid(cnfr_grad) - 1  # gradient clipping

            lr = set_lr(step, self.torsion_param, self.ligand.number_of_frames)  # set learning rate
            self.cnfr = self.cnfr - cnfr_grad * lr

            cnfr_grad.data.zero_()
            self.cnfr = self.cnfr.detach().numpy()

            self.cnfr = torch.from_numpy(self.cnfr).requires_grad_()


def run_optimization(rec_fpath: str,
            lig_fpath: str,
            native_pose_fpath: str,
            model_fpath: str,
            mean_std_file: str,
            output_path: str):

    # parse receptor
    receptor = Receptor(rec_fpath)
    receptor.parse_receptor()
    print("Receptor parsed")

    # parse ligand
    ligand = LigandConformation(lig_fpath)
    print("Ligand parsed")

    optimizer = Optimize(output_dpath=output_path,
                         receptor=receptor,
                         ligand=ligand,
                         native_pose_fpath=native_pose_fpath,
                         model_fpath=model_fpath,
                         mean_std_file=mean_std_file)
    optimizer.run()

