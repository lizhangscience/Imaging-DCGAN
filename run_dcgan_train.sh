#! /bin/bash
#PBS -S /bin/bash

#################################
## Script to run DCGAN on Cobweb
## 
##################################

## Name of job
#PBS -N plz_converge

## Queue to submit to: gpu/compute/test
#PBS -q gpu

##run parameters
#PBS -l walltime=100:00:00
#PBS -l nodes=1:v100
#PBS -l mem=50gb

##error output and path variables
#PBS -j oe
#PBS -V

cd /share/data/gordon/deep_learning/Imaging-DCGAN
##cat $PBS_NODEFILE > nodes

##loading required modules
bash
source .bashrc
module purge
module load anaconda/3-4.4.0
export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64:$LD_LIBRARY_PATH

##run job
python train-dcgan.py --dataset data/normalised/HST_aug_data_6000.npy --epoch 100 --batch_size 50 --train_size 2100
#python complete.py --imgSize 64 --dataset data/normalised/normalised_cleaned_sasha_test.npy --batch_size 1 --nIter 2000 --train_size 70
