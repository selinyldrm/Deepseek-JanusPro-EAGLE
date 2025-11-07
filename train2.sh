#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=ds-eagle         # Name of the job
#SBATCH --output=train-%j.log            # Output log file
#SBATCH --error=train-%j.log              # Error log file
#SBATCH --time=4:00:00                # Max runtime (HH:MM:SS)
#SBATCH --partition=mi3258x          # Partition to submit to
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=seliny2@illinois.edu

# sed -i 's/state_5\/model.safetensors/state_10\/model.safetensors/' /work1/deming/seliny2/LANTERN/traineagle3/main.py

source /home1/seliny2/.bashrc
conda activate lantern
cd /work1/deming/seliny2/LANTERN
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd traineagle3
./run_training_300s.sh
# python3 main.py generate_train_data --model llamagen2  --data_path /work1/deming/shared/llamagen/train2017-tokenized --output_dir /work1/deming/shared/llamagen/selin-eagle3-train-data_eagle3  --num_samples 118287

