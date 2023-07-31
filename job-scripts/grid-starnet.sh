#!/bin/bash
#SBATCH --mem=1G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=168:0:0
#SBATCH --mail-user=wiltonfs@student.ubc.ca
#SBATCH --mail-type=ALL

cd $PROJECT/astro_research

# I want to try the following combos:
# --ns = [0, 0.3, 0.5]
# --bs = [16, 128, 1024]
# --lr = [0.0001, 0.001, 0.005]

# Loop through the combinations and pass them to grid-sub-script.sh
for ns_value in 0 0.3 0.5; do
  for bs_value in 16 128 1024; do
    for lr_value in 0.0001 0.001 0.005; do
      sbatch job-scripts/grid-sub-script.sh $ns_value $bs_value $lr_value
    done
  done
done


