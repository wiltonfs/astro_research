#!/bin/bash
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=3:00:0
#SBATCH --mail-user=wiltonfs@student.ubc.ca
#SBATCH --mail-type=ALL

# Check if the required number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <bs_value> <i_value>"
    exit 1
fi

cd $PROJECT/astro_research
module purge
module load python scipy-stack
source ~/astroPy/bin/activate

# Access the arguments passed to the script and use them in the Python command
bs_value=$1
i_value=$2

# Train model
project_name=$(python starnet.py --id "EpochNorm/Grid" --bs $bs_value --i $i_value --vs 25  --ns 0)
# Generate visualizations
python indiv-results.py --p "$project_name"
