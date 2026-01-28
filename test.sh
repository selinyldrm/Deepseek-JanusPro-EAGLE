#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=lumina         # Name of the job
#SBATCH --output=lumina-%j.log            # Output log file
#SBATCH --error=lumina-%j.log              # Error log file
#SBATCH --time=12:00:00                # Max runtime (HH:MM:SS)
#SBATCH --partition=mi2508x          # Partition to submit to
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
# python3  main.py generate_images --model anole  --model_type eagle  --model_path /work1/deming/shared/anole/Anole-7b-v0.1-hf/  --drafter_path /work1/deming/shared/llamagen/eagle2-drafters/anole_eagle2    --output_dir /work1/deming/shared/anole/loss-scaled/lantern++-mi250  --temperature 1  --top_k 10 --top_p 1.0  --cfg 3.0 --lantern --lantern_k 10 --lantern_delta 3.0  --static_tree --tree_choice naive_extend_57 --prompt MSCOCO2017Val --num_images 5000  --multigpu 
# python3  main.py generate_images --model anole  --model_type base --model_path /work1/deming/shared/anole/Anole-7b-v0.1-hf/  --output_dir /work1/deming/shared/anole/base-mi250 --temperature 1  --top_k 10 --top_p 1.0  --cfg 3.0 --prompt MSCOCO2017Val --num_images 5000  --multigpu  
# python3  main.py generate_images --model llamagen2  --model_type eagle  --model_path /work1/deming/shared/llamagen/LlamaGen-T2I-2/ --drafter_path /work1/deming/shared/llamagen/eagle3-drafters/llamagen2-eagle3-relaion-lossscaled/llamagen2_lr0.0001_p_w0.1_bsz8_gradacc_1_epochs40_length7_mscoco2017train30k/state_22  --output_dir /work1/deming/shared/llamagen/eagle3-results/mscoco/loss-scaled/relaion-single-merge-cos0.0-kl3.0  --temperature 1  --top_k 2000 --top_p 1.0  --cfg 3.0 --lantern --lantern_k 10 --lantern_delta 3.0  --static_tree --tree_choice naive_extend_57 --multigpu --prompt MSCOCO2017Val --num_images 5000
# python3  main.py generate_images --model lumina_mgpt  --model_type eagle  --model_path /work1/deming/shared/lumina/Lumina-mGPT-7B-768/  --drafter_path /work1/deming/shared/lumina/LANTERN-Lumina-mGPT-7B-768/  --output_dir /work1/deming/shared/lumina/draft-based-interstep/intra0.9-kl3.0-inter-0.625-kl1.0  --temperature 1  --top_k 2000 --top_p 1.0  --cfg 3.0 --lantern --lantern_k 10 --lantern_delta 3.0 --static_tree --tree_choice naive_extend_57 --prompt MSCOCO2017Val --multigpu
python3  main.py generate_images --model lumina_mgpt  --model_type eagle  --model_path  /work1/deming/shared/lumina/Lumina-mGPT-7B-768/  --drafter_path /work1/deming/shared/lumina/LANTERN-Lumina-mGPT-7B-768  --output_dir /work1/deming/shared/lumina/normalized-recovered-results/0.9-0.625-kl1.0  --temperature 1  --top_k 2000 --top_p 1.0  --cfg 3.0  --static_tree --tree_choice naive_extend_57 --prompt MSCOCO2017Val  --lantern --lantern_k 10 --lantern_delta 3.0 --s_idx 4000 --e_idx 5000 --multigpu
