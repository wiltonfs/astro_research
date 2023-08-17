#!/bin/bash
#SBATCH --mem=1G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:10:0
#SBATCH --mail-user=wiltonfs@student.ubc.ca
#SBATCH --mail-type=ALL

cd $PROJECT/astro_research

# I want to try the following combos:
# --bs = [16, 128, 1024]
# --i =  [25, 200, 1600]

# Loop through the combinations and pass them to grid-sub-script.sh
for bs_value in 16 128 1024; do
  for i_value in 25 200 1600; do
    sbatch job-scripts/sub-grid-script.sh $bs_value $i_value
  done
done


