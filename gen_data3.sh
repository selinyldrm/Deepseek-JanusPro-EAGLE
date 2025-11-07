#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=train_data3       # Name of the job
#SBATCH --output=train_data3-%j.log            # Output log file
#SBATCH --error=train_data3-%j.err              # Error log file
#SBATCH --time=12:00:00                # Max runtime (HH:MM:SS)
#SBATCH --partition=mi2508x          # Partition to submit to
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=seliny2@illinois.edu

source /home1/seliny2/.bashrc
conda activate lantern
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd /work1/deming/seliny2/LANTERN
python3 main.py generate_train_data --model llamagen2 --data_path /work1/deming/shared/relaion-coco-tokenized --output_dir /work1/deming/shared/relaion-coco-training-data --eagle3 --eagle3_config /work1/deming/seliny2/LANTERN/traineagle3/config_llamagen.json --start 2200000 --end  2300000
# python3 main.py generate_train_data --model llamagen2 --data_path /work1/deming/shared/relaion-coco-tokenized --output_dir /work1/deming/shared/relaion-coco-training-data --eagle3 --eagle3_config /work1/deming/seliny2/LANTERN/traineagle3/config_llamagen.json --start 1700000 --end 1800000 