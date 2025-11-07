#!/bin/bash
#SBATCH --exclusive
#SBATCH --dependency=afterany:238357
#SBATCH --job-name=preprocess        # Name of the job
#SBATCH --output=preprocess-%j.log            # Output log file
#SBATCH --error=preprocess-%j.log              # Error log file
#SBATCH --time=12:00:00                # Max runtime (HH:MM:SS)
#SBATCH --partition=mi2104x          # Partition to submit to
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1    
#SBATCH --cpus-per-task=128     
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=seliny2@illinois.edu

source /home1/seliny2/.bashrc
conda activate lantern
cd /work1/deming/shared/llamagen/relaion-coco
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 preprocess.py