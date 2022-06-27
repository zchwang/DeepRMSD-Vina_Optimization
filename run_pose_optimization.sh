#!/bin/bash

inputs_file=inputs.dat

cat $inputs_file | while read line
do
	
	STR=($line)

	# the path of input files
	receptor_fpath=${STR[0]}
	pose_fpath=${STR[1]}
	native_pose_fpath=${STR[2]}

	INP=(${pose_fpath//// })
	target=${INP[1]}
	pose_name=${INP[3]}
	
	# the output path
	output_dpath=test_output/${target:0:4}/${pose_name:0:(-6)}

	model_fpath=models/bestmodel_cpu.pth	
	mean_std_file=models/r6-r1_0.3-2.0nm_train_mean_std.csv

	if [ -d $output_dpath ];then
		rm -r $output_dpath
	fi
	
	python scripts/run.py -receptor	$receptor_fpath -pose $pose_fpath -native_pose $native_pose_fpath -model $model_fpath -mean_std_file $mean_std_file -output_path $output_dpath

done


