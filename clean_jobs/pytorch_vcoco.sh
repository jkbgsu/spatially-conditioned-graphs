#!/bin/bash
#SBATCH --ntasks-per-node=1
#####SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=2
#SBATCH --output=7_25/vcoco_test_sum_stdout_07-15-23_%j.txt
#SBATCH --error=7_25/vcoco_test_sum_error_07-15-23_%j.txt
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --job-name=vcoco_simple_comb
#SBATCH --account=PCS0273


#salloc test: salloc --account=PCS0273 --cpus-per-task=1 --ntasks-per-node=1 --gpus-per-node=1 --time=01:00:00 srun --pty /bin/bash
module load python/3.7-2019.10 cuda/11.1.1

echo "loaded needed mods from head"
which python

pwd
which conda
python -c "import torch; print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))"
source activate base #pytorch_7

cd /fs/scratch/PCS0273/jkblank/jkbgsusc/repos/SCG-JB/spatially-conditioned-graphs/
pwd
python -c "import torch; print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))"
nvcc --version
python main.py --world-size 2 --dataset vcoco --partitions trainval val --data-root vcoco --train-detection-dir vcoco/detections/trainval_emotions_avg --val-detection-dir vcoco/detections/trainval_emotions_avg --print-interval 20 --cache-dir checkpoints/vcoco_7_25_sum/