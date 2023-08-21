#!/bin/bash

cd /home/wiltonfs/scratch/wiltonfs/astro_research
module purge
module load python scipy-stack
source ~/astroPy/bin/activate

# Train model
project_name=$(python starnet.py --i 1000 --vs 25 --ns 0 | grep -oP '\$([^$]+)\$')
project_name=${project_name:1:-1}
# Generate visualizations
python indiv-results.py --p "$project_name"
