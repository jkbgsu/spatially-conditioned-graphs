#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:10:00
#SBATCH --job-name=mmlab_example
#SBATCH --account=PCS0273

# module load python/3.9

cd ~
pwd
pip install deepface
