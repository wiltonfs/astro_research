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

# weights and biases
pip install --no-index wandb
wandb login $WAND_API_KEY 
wandb offline

# Train model
project_name=$(python starnet.py --i 1000 --vs 25 --ns 0 | grep -oP '\$([^$]+)\$')
project_name=${project_name:1:-1}
# Generate visualizations
python indiv-results.py --p "$project_name"
