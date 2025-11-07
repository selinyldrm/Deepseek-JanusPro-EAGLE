#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=tokenize5       # Name of the job
#SBATCH --output=tokenize5-%j.log            # Output log file
#SBATCH --error=tokenize5-%j.txt              # Error log file
#SBATCH --time=12:00:00                # Max runtime (HH:MM:SS)
#SBATCH --partition=mi2508x          # Partition to submit to
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=seliny2@illinois.edu

source /home1/seliny2/.bashrc
conda activate lantern
cd /work1/deming/seliny2/LANTERN
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 main.py extract_code --model llamagen2 --data_path /work1/deming/shared/relaion-coco --output_dir /work1/deming/shared/relaion-coco-tokenized --start 900000  --end 1000000