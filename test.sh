#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=anole-5k        # Name of the job
#SBATCH --output=anole-5k-%j.log            # Output log file
#SBATCH --error=anole-5k-%j.log              # Error log file
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
# accelerate launch -m entrypoints.train_drafter.main --model llamagen2  --base_path ../../shared/llamagen/LlamaGen-T2I-2/  --config_path ./data/configs/llamagen_t2i2_config.json  --data_dir ../../shared/llamagen/training-data/  --save_dir ../../shared/llamagen/trained-model-temp-loss4x
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3  main.py generate_images --model anole  --model_type eagle  --model_path /work1/deming/shared/anole/Anole-7b-v0.1-hf/  --drafter_path /work1/deming/seliny2/LANTERN/entrypoints/train_drafter/anole_drafter/   --output_dir ./../shared/anole/lantern++-t1-topk2000-topp1.0-cfg3-lanternk10-cfg3  --temperature 1  --top_k 2000 --top_p 1.0  --cfg 3.0 --lantern --lantern_k 10 --lantern_delta 3.0  --static_tree --tree_choice naive_extend_57 --prompt MSCOCO2017Val --num_images 5000 --multigpu 