#!/bin/bash
#SBATCH --ntasks-per-node=1
#####SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --output=results/vcoco_pretrain_hicodet_test_stdout_07-15-23_%j.txt
#SBATCH --error=results/vcoco_pretrain_hicodet_test_error_07-15-23_%j.txt
#SBATCH --mem=50G
#SBATCH --time=01:00:00
#SBATCH --job-name=test_torch_hicodet
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
#python main.py --world-size 4 --human-emotion True --cache-dir checkpoints/hicodet_7_23/
python test.py --model-path checkpoints/vcoco_7_23/ckpt_02177_07.pt