#!/bin/bash
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=3:00:0

# Check if the required number of arguments are provided
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <ID> <bs> <lrI> <lrF> <i> <ns>"
    exit 1
fi

cd $SCRATCH/astro_research
module purge
module load python scipy-stack
source ~/astroPy/bin/activate

# weights and biases
pip install --no-index wandb
wandb login $WAND_API_KEY 
wandb offline

# Access the arguments passed to the script and use them in the Python command
ID=$1
bs=$2
lrI=$3
lrF=$4
i=$5
ns=$6

# Train model
project_name=$(python starnet.py --id $ID --bs $bs --lrI $lrI --lrF $lrF --i $i --vs 25  --ns $ns | grep -oP '\$([^$]+)\$')
project_name=${project_name:1:-1}
# Generate visualizations
python starnet-analysis.py --p "$project_name"
