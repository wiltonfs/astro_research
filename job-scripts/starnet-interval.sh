#!/bin/bash
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=3:00:0

cd $SCRATCH/astro_research
module purge
module load python scipy-stack
source ~/astroPy/bin/activate

# Train model
python starnet-intervals.py
