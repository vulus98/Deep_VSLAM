#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=50G
source /itet-stor/vbozic/net_scratch/conda/etc/profile.d/conda.sh
source activate deep_vslam
python -u /srv/beegfs02/scratch/deep_slam/data/vukasin/semester_project/deep-vslam/train_network.py "$@"
