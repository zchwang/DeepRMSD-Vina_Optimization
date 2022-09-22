#!/bin/bash

# Usage:
# bash run_pose_optimization.sh inputs.dat 

inputs_file=$1

cat $inputs_file | while read line
do
	
	STR=($line)

	# the path of input files
	code=${STR[0]}
	receptor_fpath=${STR[1]}
	poses_dpath=${STR[2]}

	
	# the output path
	output_dpath=test_output/${code}

	model_fpath=models/bestmodel_cpu.pth	
	mean_std_file=models/r6-r1_0.3-2.0nm_train_mean_std.csv

	if [ -d $output_dpath ];then
		rm -r $output_dpath
	fi
	
	python scripts/run.py \
	       -receptor $receptor_fpath \
	       -poses_dpath $poses_dpath \
	       -model $model_fpath \
	       -mean_std_file $mean_std_file \
	       -output_path $output_dpath

done


