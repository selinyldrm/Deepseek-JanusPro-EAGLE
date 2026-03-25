#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=tokenize       # Name of the job
#SBATCH --output=tokenize-%j.log            # Output log file
#SBATCH --error=tokenize-%j.txt              # Error log file
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
export PYTHONPATH=$PYTHONPATH:/work1/deming/seliny2/LANTERN/Janus
# python3 main.py extract_code --model lumina_mgpt --data_path /work1/deming/shared/relaion-coco --output_dir /work1/deming/shared/lumina/relaion-coco-tokenized --start 0 --end 100000
# python3 main.py extract_code --model lumina_mgpt --output_dir /work1/deming/shared/lumina/relaion-tokenized --data_path /work1/deming/shared/lumina/relaion-clean.json --start 0 --end 100000 --num_samples 500000
python3 main.py extract_code --model januspro --output_dir /work1/deming/shared/Janus-Pro-Training-Data  --data_path /work1/deming/shared/llamagen/mscoco_train_dataset/images --start 30000 --end 60000 