#!/bin/bash
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=3:00:0
#SBATCH --mail-user=wiltonfs@student.ubc.ca
#SBATCH --mail-type=ALL

cd $PROJECT/astro_research
module purge
module load python scipy-stack
source ~/astroPy/bin/activate

# Train model
project_name=$(python starnet.py --i 100 --vs 5 --ns 0 | grep -oP '\$([^$]+)\$')
project_name=${project_name:1:-1}
# Generate visualizations
python indiv-results.py --p "$project_name"
