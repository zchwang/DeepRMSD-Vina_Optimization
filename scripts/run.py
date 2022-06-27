from argparse import RawDescriptionHelpFormatter
import argparse
from argparse_utils import add_args
from optimization import run_optimization
import datetime
from model import CNN

d = """
    Update the ligand conformation ...
    """

parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
add_args(parser)
args = parser.parse_args()

receptor_fpath = args.receptor
pose_fpath = args.pose
native_pose_path = args.native_pose
model_fpath = args.model
mean_std_file = args.mean_std_file
output_path = args.output_path

def main():

    print("Start Now ...")

    run_optimization(rec_fpath=receptor_fpath,
            lig_fpath = pose_fpath,
            native_pose_fpath = native_pose_path,
            model_fpath = model_fpath,
            mean_std_file = mean_std_file,
            output_path = output_path)

start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
main()
end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

with open(output_path + '/time_running.dat', 'w') as f:
    f.writelines('Start Time:  ' + start_time + '\n')
    f.writelines('End Time:  ' + end_time)
