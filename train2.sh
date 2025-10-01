#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=train-eagle         # Name of the job
#SBATCH --output=train-%j.log            # Output log file
#SBATCH --error=train-%j.log              # Error log file
#SBATCH --time=6:00:00                # Max runtime (HH:MM:SS)
#SBATCH --partition=mi3008x          # Partition to submit to
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=seliny2@illinois.edu

source /home1/seliny2/.bashrc
conda activate lantern
cd /work1/deming/seliny2/LANTERN
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd traineagle3
sed -i 's/state_5\/model.safetensors/state_10\/model.safetensors/' /work1/deming/seliny2/LANTERN/traineagle3/main.py
./run_training.sh

