#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=train-lumina         # Name of the job
#SBATCH --output=train-lumina%j.log            # Output log file
#SBATCH --error=train-lumina%j.log              # Error log file
#SBATCH --time=12:00:00                # Max runtime (HH:MM:SS)
#SBATCH --partition=mi3258x           # Partition to submit to
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=seliny2@illinois.edu

source /home1/seliny2/.bashrc
conda activate lantern
cd /work1/deming/seliny2/LANTERN
# accelerate launch -m entrypoints.train_drafter.main --model llamagen2  --base_path ../../shared/llamagen/LlamaGen-T2I-2/  --config_path ./data/configs/llamagen_t2i2_config.json  --data_dir ../../shared/llamagen/training-data/  --save_dir ../../shared/llamagen/trained-model-temp-loss4x
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# cd traineagle3
# ./run_training.sh
# accelerate launch -m entrypoints.train_drafter.main --model llamagen2 --base_path  /work1/deming/shared/llamagen/LlamaGen-T2I-2 --config_path /work1/deming/seliny2/LANTERN/data/configs/llamagen_t2i2_config.json   --data_dir /work1/deming/shared/llamagen/training-data  --save_dir /work1/deming/shared/llamagen/llamagen2-eagle3-fixedbase-fixedconfig-fixedds-length1 --bs 8 --save_freq 2  
python3 main.py train_drafter --model lumina_mgpt --base_path /work1/deming/shared/lumina/Lumina-mGPT-7B-768 --data_dir /work1/deming/shared/lumina/training-data --save_dir /work1/deming/shared/lumina/drafter-lossscaled --coupled