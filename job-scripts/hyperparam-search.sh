#!/bin/bash
#SBATCH --mem=1G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:10:0

cd $SCRATCH/astro_research

# Hyperparameter search variables
num_searches=50

batch_range=(16 1024)
iter_range=(100 3000)
lr_initial_range=(0.01 0.0001)
lr_final_range=(0.001 0.00001)
ns_range=(0.0 0.3)

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
  ID="EpochTune/Grid" #Name of this hyperparameter search (Used as folder name)
  bs=$(random_int ${batch_range[0]} ${batch_range[1]})
  lrI=0.006  #lrI=$(random_float ${lr_initial_range[0]} ${lr_initial_range[1]})
  lrF=0.0005 #lrF=$(random_float ${lr_final_range[0]} ${lr_final_range[1]})
  I=$(random_int ${iter_range[0]} ${iter_range[1]})
  ns=0.0     #ns=$(random_float ${ns_range[0]} ${ns_range[1]})

  echo "Iteration $i:"
  echo "batch size = $bs"
  echo "inital learning rate = $lrI"
  echo "final learning rate = $lrF"
  echo "iterations = $I"
  echo "noise std = $ns"
  echo "---------------"
  sbatch job-scripts/sub-search-script.sh $ID $bs $lrI $lrF $I $ns
done



