#!/bin/bash
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=3:00:0
#SBATCH --mail-user=wiltonfs@student.ubc.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/wiltonfs/scratch/wiltonfs/astro_research/outputs/slurm

cd $SCRATCH/astro_research
module purge
module load python scipy-stack
source ~/astroPy/bin/activate

# Train model
python starnet-interval.py
