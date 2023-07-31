#!/bin/bash
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=3:00:0
#SBATCH --mail-user=wiltonfs@student.ubc.ca
#SBATCH --mail-type=ALL

# Check if the required number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <ns_value> <bs_value> <lr_value>"
    exit 1
fi

cd $PROJECT/astro_research
module purge
module load python scipy-stack
source ~/astroPy/bin/activate

# Access the arguments passed to the script and use them in the Python command
ns_value=$1
bs_value=$2
lr_value=$3

python starnet.py --id "Grid" --i 100000 --vs 25 --ns $ns_value --bs $bs_value --lr $lr_value
