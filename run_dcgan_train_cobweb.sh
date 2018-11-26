#! /bin/bash
#PBS -S /bin/bash

#################################
## Script to run DCGAN training
##################################

## Name of job
#PBS -N ImgDCGAN

## Queue to submit to: gpu/compute/test
#PBS -q gpu

##run parameters
#PBS -l walltime=50:00:00
#PBS -l nodes=1:v100
#PBS -l mem=10gb

##error output and path variables
#PBS -j oe
#PBS -V

export LD_LIBRARY_PATH=/usr/local/cuda-9.1/lib64:$LD_LIBRARY_PATH

##run job
python train-dcgan.py --dataset data/normalised/normalised_cleaned_sasha_train.npy --epoch 35 --batch_size 50 --train_size 2100
