#!/bin/bash
#SBATCH  --output=sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=50G
source /itet-stor/vbozic/net_scratch/conda/etc/profile.d/conda.sh
source activate dense_matching
python -u /srv/beegfs02/scratch/deep_slam/data/vukasin/semester_project/Hierarchical-Localization/extract_bundle_adjustment_data.py "$@"
