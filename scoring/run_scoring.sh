#!/bin/bash

if [ $# -ne 3 ];then
	echo "Usage: bash run_deeprmsd_score.sh rec_fpath pose_fpath out_fpath "
	echo "rec_fpath: .pdbqt file."
	echo "pose_fpath: .pdbqt file, which can contain multiple decoys. "
	echo "out_fpath: .csv file, which can contain the DeepRMSD, inter-vina, DeepRMSD+Vina score. "
	exit 0
fi

rec_fpath=$1
pose_fpath=$2
out_fpath=$3

model_dpath=../models

mean_std_file=${model_dpath}/r6-r1_0.3-2.0nm_train_mean_std.csv
model=${model_dpath}/bestmodel_cpu.pth

python scripts/run.py \
	-rec_fpath $rec_fpath \
	-pose_fpath $pose_fpath \
	-mean_std_file $mean_std_file \
	-model $model \
	-out_fpath $out_fpath

