#!/bin/sh

export PYTHONPATH=./
eval "$(conda shell.bash hook)"
eval "conda activate pt"
PYTHON=python


TRAIN_CODE=train.py
TEST_CODE=test.py

dataset=$1
exp_name=$2
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}.yaml

mkdir -p ${model_dir} ${result_dir}
mkdir -p ${result_dir}/last
mkdir -p ${result_dir}/best
cp tool/train.sh tool/${TRAIN_CODE} ${config} tool/test.sh tool/${TEST_CODE} ${exp_dir}


now=$(date +"%Y%m%d_%H%M%S")
echo "$($PYTHON -c "import sys; print(sys.version)")"
echo $PYTHON ${exp_dir}/${TRAIN_CODE} \
  --config=${config} \
  save_path ${exp_dir} \
  2>&1 | tee ${exp_dir}/train-$now.log
  
$PYTHON ${exp_dir}/${TRAIN_CODE} \
  --config=${config} \
  save_path ${exp_dir} \
  2>&1 | tee ${exp_dir}/train-$now.log

#CUDA_LAUNCH_BLOCKING=1 $PYTHON /home/tidop/Documents/git/CloudClassifier-DL/training.py --train /home/tidop/Downloads/olloki_TF/olloki_substation_trainV22_5cm_sub.txt --test /home/tidop/Downloads/olloki_TF/traintest_features_5cm_sub.txt --features 10,11,16  --labels 6 --output /home/tidop/Downloads/olloki_TF/output1 --epoch 60

#CUDA_LAUNCH_BLOCKING=1 $PYTHON /home/tidop/Documents/git/CloudClassifier-DL/training.py --train /home/tidop/Downloads/InRoad_TF/train_inroadV4.txt --test /home/tidop/Downloads/InRoad_TF/test_inroadV4.txt --features 11 --labels 12 --output /home/tidop/Downloads/InRoad_TF/output1 --epoch 60
#10,11,16

#CUDA_LAUNCH_BLOCKING=1 $PYTHON /home/tidop/Documents/git/CloudClassifier-DL/training.py --train /home/tidop/Downloads/railway_TF/railway_trainV2.txt --test /home/tidop/Downloads/railway_TF/railway_testV2.txt  --labels 21 --output /home/tidop/Downloads/railway_TF/output1 --epoch 60

#CUDA_LAUNCH_BLOCKING=1 $PYTHON /home/tidop/Documents/git/CloudClassifier-DL/training_supercomputation_S3DIS_pt_repo.py --train None --test None  --labels 21 --output /home/tidop/Downloads/s3dis_TF/output1 --epoch 60

#version DDP
#CUDA_LAUNCH_BLOCKING=1 $PYTHON /home/tidop/Documents/git/CloudClassifier-DL/training.py --train /home/tidop/Downloads/olloki_TF/olloki_substation_trainV22_5cm_sub.txt --test /home/tidop/Downloads/olloki_TF/traintest_features_5cm_sub.txt --features 10,11,16  --labels 6 --output /home/tidop/Downloads/olloki_TF/output1 --epoch 6

#version avec cloud classifier DL
#CUDA_LAUNCH_BLOCKING=1 $PYTHON /home/tidop/Documents/git/CloudClassifier-DL/training.py --train "/home/tidop/Downloads/nubes de puntos/PT_S3DIS_pt_files/train" --test "/home/tidop/Downloads/nubes de puntos/PT_S3DIS_pt_files/val" --features 10,11,16  --labels 6 --output /home/tidop/Downloads/olloki_TF/output1 --epoch 60

