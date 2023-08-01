#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=2
#SBATCH --output=7_29_eql_stressed/hicodet_test_eql_stres_stdout_07-29-23_%j.txt
#SBATCH --error=7_29_eql_stressed/hicodet_test_eql_stres_error_07-29-23_%j.txt
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --job-name=hicodet_simple_comb
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
python main.py --world-size 2 --human-emotion True --cache-dir checkpoints/7_29_eql_stressed/ --num-iter 2
