#!/bin/bash
#SBATCH --output=log/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=30G

# source /itet-stor/petars/net_scratch/conda/etc/profile.d/conda.sh
# source /scratch_net/biwidl206/petars/conda//etc/profile.d/conda.sh

#conda activate dense_matching_env
/home/nipopovic/scratch_second/apps/miniconda3/envs/dense_matching_env/bin/python -u extract_bundle_adjustment_data.py "$@"