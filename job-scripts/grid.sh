#!/bin/bash
#SBATCH --mem=1G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:10:0
#SBATCH --mail-user=wiltonfs@student.ubc.ca
#SBATCH --mail-type=ALL

cd $PROJECT/astro_research

# Define variables
num_searches=30

batch_range=(16 1024)
lr_initial_range=(0.01 0.0001)
lr_final_range=(0.001 0.00001)

# Define function to generate random integer in a range
function random_int() {
  echo $((RANDOM % ($2 - $1 + 1) + $1))
}

# Define function to generate random float in a range
function random_float() {
  local lower=$1
  local upper=$2
  range_diff=$(echo "$upper - $lower" | bc)
  rand_decimal=$(echo "scale=10; $RANDOM/32767" | bc)
  echo "$(echo "scale=10; $lower + $rand_decimal * $range_diff" | bc)"
}

# Loop for num_searches iterations
for ((i=1; i<=$num_searches; i++)); do
  bs_value=$(random_int ${batch_range[0]} ${batch_range[1]})
  lrI_value=$(random_float ${lr_initial_range[0]} ${lr_initial_range[1]})
  lrF_value=$(random_float ${lr_final_range[0]} ${lr_final_range[1]})
  echo "Iteration $i:"
  echo "bs_value = $bs_value"
  echo "lrI_value = $lrI_value"
  echo "lrF_value = $lrF_value"
  echo "------"
  sbatch job-scripts/sub-grid-script.sh $bs_value $lrI_value $lrF_value
done



