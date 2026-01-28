#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=datagen       # Name of the job
#SBATCH --output=datagen-%j.log            # Output log file
#SBATCH --error=datagen-%j.err              # Error log file
#SBATCH --time=4:00:00                # Max runtime (HH:MM:SS)
#SBATCH --partition=mi2104x          # Partition to submit to
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=seliny2@illinois.edu

source /home1/seliny2/.bashrc
conda activate lantern
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd /work1/deming/seliny2/LANTERN
# python3 main.py generate_train_data --model llamagen2 --data_path /work1/deming/shared/relaion-coco-tokenized --output_dir /work1/deming/shared/relaion-coco-training-data --eagle3 --eagle3_config /work1/deming/seliny2/LANTERN/traineagle3/config_llamagen.json --start 1981205 --end 2000000
# python3 main.py generate_train_data --model llamagen2 --data_path /work1/deming/shared/relaion-coco-tokenized --output_dir /work1/deming/shared/relaion-coco-training-data --eagle3 --eagle3_config /work1/deming/seliny2/LANTERN/traineagle3/config_llamagen.json --start 1500000 --end 1600000
python3 main.py generate_train_data --model lumina_mgpt --data_path /work1/deming/shared/lumina/relaion-tokenized --output_dir /work1/deming/shared/lumina/lumina-training-data  --start 440000 --end 450000