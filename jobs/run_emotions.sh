#!/bin/bash
#SBATCH --ntasks-per-node=1
#####SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --output=vcoco_pretrain_hicodet_test_logs/hicodet_test_stdout_07-15-23_%j.txt
#SBATCH --error=vcoco_pretrain_hicodet_test_logs/hicodet_test_error_07-15-23_%j.txt
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --job-name=hicodet_emotion_SCG
#SBATCH --account=PCS0273

module load python/3.7-2019.10 cuda/11.1.1
#module load python/3.7-2019.10 cuda/11.1.1 
#module load cuda/11.1.1
#module load nccl/2.11.4
#module load cp2k/2022.2
#module load cuda/10.2.89
echo "loaded needed mods from head"
which python
###conda init bash
###. ~/.bashrc
###source /apps/python/3.7-2019.10/bin/conda
pwd
which conda
python -c "import torch; print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))"
#source activate pytorch_2
source activate pytorch_7
#source activate pocket_batch NOTE: NG
#conda create -n pocket_ncclTry_118 python=3.8 -y
#source activate pocket_ncclTry_118
#conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
#conda install pytorch torchvision cudatoolkit=10.2 -c pytorch -y

#onda install matplotlib tqdm scipy -y
#source activate base
#echo "activate base env"
#source activate pocket_batch
#echo "activated pocket env"
#conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

#cd /fs/scratch/PCS0273/jkblank/jkbgsusc/repos/pocket-master/

#pip install -e .
cd /fs/scratch/PCS0273/jkblank/jkbgsusc/repos/SCG-JB/spatially-conditioned-graphs/
pwd
python -c "import torch; print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))"
nvcc --version
CUDA_LAUNCH_BLOCKING=1 python main.py --world-size 4 --human-emotion True --cache-dir checkpoints/hicodet/


