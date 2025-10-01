<<<<<<< HEAD
import re
import os
import json, csv
import random
import argparse
import traceback

import torch
import numpy as np

import models.drafters.choices as choices
import matplotlib.pyplot as plt

from itertools import product
from torchvision.utils import save_image
import seaborn as sns
import torch.nn.functional as F
from tqdm import tqdm

USE_EXPERIMENTAL_FEATURES = os.getenv("USE_EXPERIMENTAL_FEATURES", "0") == "1"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model to use for image generation",
                        default="lumina_mgpt")
    parser.add_argument("--test", action="store_true", help="Use LANTERN for debugging")
    parser.add_argument("--relaxed", action="store_true", help="Use LANTERN for debugging")
    parser.add_argument("--multigpu", action="store_true", help="Use LANTERN to scale")
    parser.add_argument("--model_type", type=str, help="Model type; choices: ['eagle', 'base', 'vllm']",
                        default="eagle")
    parser.add_argument("--model_path", type=str, help="Path to the model",
                        default="Alpha-VLLM/Lumina-mGPT-7B-768")
    parser.add_argument("--drafter_path", type=str, help="Path to the drafter model",
                        default="ckpts/lumina_mgpt/trained_drafters/lumina_mgpt_7b_768_lr0.0001_p_w0.1"
                        "_bsz2_gradacc_8_epochs20_noise_uniform_std0.2_coupled_False_cfgloss_False"
                        "_cfgscale_3.0_embed_upscale1.0_mscoco2017train30k/state_20")
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--target_size", type=int, default=768)

    parser.add_argument("--prompt", type=str, help="Prompt for image generation",
                        default="PartiPrompts")
    parser.add_argument("--num_images", type=int, help="Number of images to generate",
                        default=10)
    parser.add_argument("--slice", type=str, help="Slice of prompts to use; format: 'start-end'",
                        default=None)
    parser.add_argument("--output_dir", type=str, help="Output directory for generated images",
                        default="generated_images")
    
    parser.add_argument("--set_seed", action="store_true", help="Set random seed for reproducibility")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation")
    parser.add_argument("--top_k", type=int, default=2000, help="Top-k for generation")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for generation")
    parser.add_argument("--cfg", type=float, default=3.0, help="CFG for generation")
    
    # LANTERN-specific arguments
    parser.add_argument("--lantern", action="store_true", help="Use LANTERN for image generation")
    parser.add_argument("--lantern_k", type=int, default="1000", help="Value of k for LANTERN")
    parser.add_argument("--lantern_delta", type=float, default="0.1", help="Value of delta for LANTERN")
    parser.add_argument("--grid_search", action="store_true", help="Run grid search for LANTERN hyperparameters")

    parser.add_argument("--static_tree", action="store_true", help="Use static tree based drafting")
    parser.add_argument("--eagle_version", type=int, default=1, help="EAGLE version")
    parser.add_argument("--cfg_mode", type=str, default="sequential", help="CFG mode")

    # Experimental arguments
    parser.add_argument("--tree_choices", type=str, help="Tree choice for LANTERN",
                        default="mc_sim_7b_63")
    parser.add_argument("--drafter_top_k", type=int, default=None, help="Top-k for drafter")

    # legacy arguments
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for image generation")
    parser.add_argument("--end_idx", type=int, default=10000, help="End index for image generation")

    return parser

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

def load_model(args):
    if args.model == "lumina_mgpt":
        if args.model_type == "vllm":
            from models.base_models.lumina_mgpt.vllm_inference_solver import FlexARInferenceSolver
            model = FlexARInferenceSolver(
                model_path=args.model_path,
                precision=args.precision,
                target_size=args.target_size,
                max_num_seqs=24,
            )

        elif args.model_type == "base":
            from models.base_models.lumina_mgpt.inference_solver import FlexARInferenceSolver
            model = FlexARInferenceSolver(
                model_path=args.model_path,
                precision=args.precision,
                target_size=args.target_size,
                cfg_mode=args.cfg_mode,
            )

        elif args.model_type == "eagle":
            from models.base_models.lumina_mgpt.eagle_inference_solver import FlexARInferenceSolver
            model = FlexARInferenceSolver(
                model_path=args.model_path,
                drafter_path=args.drafter_path,
                precision=args.precision,
                target_size=args.target_size,
                cfg_mode=args.cfg_mode,
                eagle_version=args.eagle_version,
            )
        else:
            raise ValueError(f"Model type {args.model_type} is not supported for model {args.model}")
    elif args.model == "anole":
        if args.model_type == "vllm":
            raise NotImplementedError("VLLM model is not supported for Anole model")
        
        elif args.model_type == 'base':
            from models.kv_variants.modeling_anole_kv import ChameleonForConditionalGeneration
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
            model = ChameleonForConditionalGeneration.from_pretrained(args.model_path).to(dtype=dtype, device='cuda')
            model.eval()
        
        elif args.model_type == 'eagle':
            from models.ea_model_anole import EaModel
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
            model = EaModel.from_pretrained(base_model_path=args.model_path, ea_model_path=args.drafter_path).to(dtype=dtype, device='cuda')
            model.eval()
        
        else:
            raise ValueError(f"Model type {args.model_type} is not supported for model {args.model}")
    elif "llamagen" in args.model:
        if args.model_type == "vllm":
            raise NotImplementedError("VLLM model is not supported for LLAMAGen model")
        
        elif args.model_type == 'base':
            from models.kv_variants.modeling_llamagen_kv import LlamaForCausalLM
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
            # dtype = torch.float32
            model = LlamaForCausalLM.from_pretrained(args.model_path).to(dtype=dtype, device='cuda')
            model.eval()
        
        elif args.model_type == 'eagle':
            from models.ea_model_llamagen import EaModel
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
            # dtype = torch.float32
            model = EaModel.from_pretrained(base_model_path=args.model_path, ea_model_path=args.drafter_path).to(dtype=dtype, device='cuda')
            model.eval()
        else:
            raise ValueError(f"Model type {args.model_type} is not supported for model {args.model}")
    else:
        raise NotImplementedError(f"Model {args.model} is not supported")
    
    return model

def load_prompts(args):
    prompts = []
    if args.prompt == "PartiPrompts":
        with open('data/prompts/PartiPrompts.tsv', 'r') as f:
            tsv_reader = csv.DictReader(f, delimiter='\t')
            for row in tsv_reader:
                prompts.append(row['Prompt'])
    elif args.prompt == "MSCOCO2017Val":
        with open('data/prompts/captions_val2017_longest.json', 'r') as f:
            captions = json.load(f)
            for caption in captions:
                prompts.append(caption)
    

    # if args.slice is not None:
    #     assert re.match(r'^\d+-\d+$', args.slice), f"Invalid format: '{args.slice}'. Expected format is 'start-end'."

    #     start, end = map(int, args.slice.split('-'))
    #     assert start < end, f"Invalid range: '{args.slice}'. Start value must be less than end value."
    #     assert start >= 0 and end >= 0, "Slice values must be non-negative."

    #     prompts = prompts[start:end]
    
    # if args.num_images < len(prompts):
    #     print(f"Number of images to generate is less than the number of prompts. Sampling {args.num_images} prompts.")
    #     prompts = random.sample(prompts, args.num_images)
    # else:
    #     print(f"Number of images to generate is greater than the number of prompts. Generating only {len(prompts)} images and no sampling.")
    #     pass

    # if args.multigpu:
    #     with open("/work1/deming/seliny2/LANTERN/global_statistics_0_100.json", "r") as f:
    #         data = json.load(f)
    # else:
    #     with open("/work1/deming/seliny2/LANTERN/global_statistics_0_5.json", "r") as f:
    #         data = json.load(f)
        
    # # # Extract prompt fields 
    # prompts = [entry["prompt"] for entry in data.values()]
    
    # if args.multigpu :
    #     return prompts[:100]
    
    # delete below before return
    # with open("/mnt/shared/gpfs/home/seliny2/LANTERN/global_statistics_0_100.json", "r") as f:
    #     data = json.load(f)
    # prompts = [entry["prompt"] for entry in data.values()]
    
    return prompts

def generate_and_save_image(output_dir, model, model_name, prompt, img_save_path, test, relaxed, **kwargs):
    # print(f"Generating image for prompt: {prompt}")
    if model_name == "lumina_mgpt":
        generate_params = {
            "images": [],
            "qas": [[prompt, None]],
            "max_gen_len": 2354,
            "temperature": kwargs["temperature"],
            "top_k": kwargs["top_k"],
            "cfg_scale": kwargs["cfg"],
        }
    elif model_name in ["anole", "llamagen", "llamagen2"]:
        if model_name == "llamagen":
            max_gen_len = 256
        else:
            max_gen_len = 1024
        generate_params = {
            "prompt": [prompt],
            "max_length": max_gen_len,
            "temperature": kwargs["temperature"],
            "top_k": kwargs["top_k"],
            "top_p": kwargs["top_p"],
            "cfg": kwargs["cfg"],
            "static_tree": kwargs["static_tree"],
        }
    else:
        raise NotImplementedError(f"Model {model_name} is not supported for image generation.")
    
    if "lantern" in kwargs:
        generate_params["lantern"] = kwargs.get("lantern", False)
        generate_params["lantern_k"] = kwargs.get("lantern_k", 1000)
        generate_params["lantern_delta"] = kwargs.get("lantern_delta", 0.1)

    if USE_EXPERIMENTAL_FEATURES:
        generate_params["tree_choices"] = kwargs["tree_choices"]
        generate_params["drafter_top_k"] = kwargs["drafter_top_k"]

    if test:
        generate_params["testing"] = True
    if relaxed:
        generate_params["relaxed"] = True
    if test:
        generated_tokens, latency, acceptance_list, analysis_p, analysis_p_p, analysis_r, overhead_list, logit_list, sim_list  = model.generate(**generate_params)
    else: 
        generated_tokens, latency = model.generate(**generate_params)
        print(f"generate time={latency} seconds")
    _, generated_image = model.decode_ids(generated_tokens)
        
    def sanitize_filename(text, max_len=100):
        # Remove unsafe characters and trim long prompts
        text = re.sub(r'[\/:*?"<>|]', '', text).strip().replace(' ', '_')
        return text[:max_len]

    if model_name in ["lumina_mgpt", "anole"]:
        generated_image[0].save(img_save_path, "png")
    elif "llamagen" in model_name:
        os.makedirs(f"{output_dir}", exist_ok=True)
        filename = sanitize_filename(prompt)
        save_image(generated_image, f"{output_dir}/{filename}.png", normalize=True, value_range=(-1, 1))
    if test:
        return latency, acceptance_list, analysis_p, analysis_p_p, analysis_r, overhead_list, logit_list, sim_list
    return latency

def run_generate_image(args):
    assert args.model_type != "vllm", "VLLM model is not supported for single image generation"

    if args.set_seed:
        set_seed(args.random_seed)
    
    prompts = load_prompts(args)
    # global_statistics = {}  
    # for idx, prompt in tqdm(enumerate(prompts), total=len(prompts)):
    #     statistics = {
    #             "prompt": prompt,
    #         }
    #     global_statistics[f"prompt_{idx}"] = statistics
    # with open(f"{args.output_dir}/global_statistics_{args.start_idx}_{args.end_idx}.json", "w") as f:
    #     json.dump(global_statistics, f, indent=4)
    

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.multigpu:
        batch_sz = int((len(prompts)-args.start_idx) / 8.0 + 0.5) 
        batch_start = [args.start_idx + (b_idx)*batch_sz for b_idx in range(8)]
        batch_end = [bs_idx + batch_sz for bs_idx in batch_start]
        if batch_end[-1] > len(prompts):
            batch_end[-1] = len(prompts)
        import torch.multiprocessing as mp
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

        mp.spawn(worker, args=(batch_start,batch_end,args,prompts, len(prompts)), nprocs=8, join=True)
    else:
        worker(0, 0,len(prompts),args,prompts, len(prompts))    # 

def worker(rank, start_idx,end_idx,args,prompts, total_prompt_count):
    torch.cuda.set_device(f"cuda:{rank}")
    model = load_model(args)

    if args.multigpu:
        start_idx = start_idx[rank]
        end_idx = end_idx[rank]
        args.start_idx = start_idx
        args.end_idx = end_idx
        print(f"Starting worker for prompts {start_idx} to {end_idx} on Rank {rank}")
    
    global_statistics = {}  
    latencies = []
    
    if args.test:
        acceptance_lists = []
        pp = []
        p = []
        r = []
        logits = []
        overheads = []
        global_sim = torch.zeros(32, 32).to(torch.float32)
    for idx, prompt in tqdm(enumerate(prompts), total=len(prompts)):
        if idx < start_idx or idx >= end_idx:
            continue
        if args.model == "lumina_mgpt":
            q1 = f"Generate an image of 768x768 according to the following prompt:\n{prompt}"
        else:
            q1 = prompt

        generate_image_kwargs = {
            "model" : model,
            "model_name" : args.model,
            "prompt" : q1,
            "temperature" : args.temperature,
            "top_k" : args.top_k,
            "top_p" : args.top_p,
            "cfg" : args.cfg,
            "lantern": args.lantern,
            "lantern_k": args.lantern_k,
            "lantern_delta": args.lantern_delta,
            "img_save_path": f"{args.output_dir}/prompt_{idx}.png",
            "static_tree": args.static_tree,
        }

        if USE_EXPERIMENTAL_FEATURES:
            try:
                tree_choices = getattr(choices, args.tree_choices)
            except AttributeError:
                print(f"Tree choices {args.tree_choices} is not a valid choice")
                return
            
            generate_image_kwargs["tree_choices"] = tree_choices
            generate_image_kwargs["drafter_top_k"] = args.drafter_top_k
        
        generate_image_kwargs["test"] = args.test
        generate_image_kwargs["relaxed"] = args.relaxed
        
        if args.test :
            latency, acceptance_list, analysis_p, analysis_p_p, analysis_r, overhead_list, logit_list, img_sim_list = generate_and_save_image(args.output_dir, **generate_image_kwargs)
            acceptance_lists.append(acceptance_list)
            p.append(analysis_p)
            pp.append(analysis_p_p)
            r.append(analysis_r)
            overheads.append(overhead_list)
            # os.makedirs(f"{args.output_dir}/heatmap-levels/{prompt}", exist_ok=True)
            # # similarity of logits that are sampled by draft
            # for idx, tensor_x  in enumerate(img_sim_list) :
            #     plt.imshow(tensor_x.cpu().numpy(), aspect='auto', cmap='coolwarm')
            #     plt.title("Logit Similarity Heatmap per Sample Step")
            #     plt.colorbar()
            #     plt.xlabel("Lower Level")
            #     plt.ylabel("Higher Level")
            #     plt.grid(True)
            #     plt.tight_layout()
            #     plt.savefig(f"{args.output_dir}/heatmap-levels/{prompt}/sample-step-{idx}.png")
            #     plt.close()

            # # similarity of logits that are actually accepted
            # masked_logits = torch.cat(logit_list, dim=0)[:1024, :]
            # masked_logits[masked_logits == float('-inf')] = 0.0
            # masked_logits = masked_logits.to(torch.float32) 
            # normalized = F.normalize(masked_logits, dim=1, eps=1e-6).to(torch.float32)  # [1024,16k]
            # cosine_sim_matrix = torch.matmul(normalized, normalized.T) # [1024,1024]
            # os.makedirs(f"{args.output_dir}/heatmap/{prompt}", exist_ok=True)
            # avg_sim = torch.zeros(32, 32).to(torch.float32).to(cosine_sim_matrix.device)
            # for row_idx in range(cosine_sim_matrix.shape[0]//32):
            #     temp = cosine_sim_matrix[row_idx*32:row_idx*32+32, row_idx*32:row_idx*32+32]  
            #     avg_sim += temp
            #     plt.imshow(temp.cpu().numpy(), aspect='auto', cmap='coolwarm')
            #     plt.title("Logit Heatmap per Image Row")
            #     plt.colorbar()
            #     plt.xlabel("Token Index")
            #     plt.ylabel("Cosine Similarity")
            #     plt.grid(True)
            #     plt.tight_layout()
            #     plt.savefig(f"{args.output_dir}/heatmap/{prompt}/row-{row_idx}.png")
            #     plt.close()    
            # avg_sim = avg_sim/32
            # global_sim = global_sim.to(cosine_sim_matrix.device)
            # global_sim += avg_sim
            # plt.imshow(avg_sim.cpu().numpy(), aspect='auto', cmap='coolwarm')
            # plt.title("Logit Heatmap per Image")
            # plt.colorbar()
            # plt.xlabel("Token Index")
            # plt.ylabel("Cosine Similarity")
            # plt.grid(True)
            # plt.tight_layout()
            # plt.savefig(f"{args.output_dir}/heatmap/{prompt}/avg.png")
            # plt.close()
            
        else:
            latency = generate_and_save_image(args.output_dir, **generate_image_kwargs)

        latencies.append(latency)

        statistics = {
            "prompt": prompt,
            # "step_compression": step_compression,
            "latency": latency
        }

        global_statistics[f"prompt_{idx}"] = statistics
    
    # global_sim/=(end_idx-start_idx)
    # plt.imshow(global_sim.cpu().numpy(), aspect='auto', cmap='coolwarm')
    # plt.title("Logit Heatmap for 100 Images")
    # plt.colorbar()
    # plt.xlabel("Token Index")
    # plt.ylabel("Cosine Similarity")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f"{args.output_dir}/avg-logit-similarity-rank{rank}.png")
    # plt.close()


    if args.multigpu:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.distributed.init_process_group("gloo", rank=rank, world_size=8)
        # Send lengths first (in case they differ)
        local_tensor = torch.tensor(latencies)
        if local_tensor.numel() < 13:
            pad_len = 13 - local_tensor.numel()
            local_tensor = torch.cat([local_tensor, torch.full((pad_len,), -1.0)])  # use -1.0 as dummy
        # Gather from all workers
        all_gathered = [torch.zeros_like(local_tensor) for _ in range(8)]
        
        torch.distributed.all_gather(all_gathered, local_tensor)

        # Reduce at rank 0
        if rank == 0:
            global_latencies = torch.cat(all_gathered)
            print(f"Global acceptance.shape: {global_latencies.shape}")

            global_latencies = global_latencies[global_latencies >= 0]
            latencies = np.array(global_latencies.cpu())
            avg_latency = np.mean(latencies, axis=0)
            print(f"Avg latency: {avg_latency} per image, with {global_latencies.shape} images.")
            plt.scatter(range(len(latencies)), latencies, marker='o', label=f"Latency of {total_prompt_count} Images with avg={avg_latency:.2f}")
            plt.xlabel("Generation Index")
            plt.ylabel("Time (Sec)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{args.output_dir}/avg-latency.png")
            plt.close()


        if args.test: 
            # min_len_oh = min(len(lst) for lst in overheads)
            min_len = min(len(lst) for lst in acceptance_lists)
            rank_acceptance = [torch.tensor(lst[:min_len]) for lst in acceptance_lists]
            rank_acceptance = torch.stack(rank_acceptance)  # shape [N, min_len]
            # rank_oh = [torch.tensor(lst[:min_len_oh]) for lst in overheads]
            # rank_oh = torch.stack(rank_oh)
            pad_len = 800
            if min_len < pad_len:
                pad = torch.zeros(rank_acceptance.shape[0], pad_len - min_len)
                rank_acceptance = torch.cat([rank_acceptance, pad], dim=1)
            if rank_acceptance.shape[0] < 13:
                pad = torch.zeros(13-rank_acceptance.shape[0], pad_len)
                rank_acceptance = torch.cat([rank_acceptance, pad], dim=0)
            # if min_len_oh < 5000:
            #     pad = torch.zeros(rank_oh.shape[0], 5000 - min_len_oh)
            #     rank_oh = torch.cat([rank_oh, pad], dim=1)
            # if rank_oh.shape[0] < 13:
            #     pad = torch.zeros(13-rank_oh.shape[0], 5000)
            #     rank_oh = torch.cat([rank_oh, pad], dim=0)
            # print(f"Rank {rank} rank_acceptance.shape: {rank_acceptance.shape}")

            # # p 
            # pmin_len = min(len(lst) for lst in p if len(lst) > 0)
            # # print(f"Rank {rank} rankp.min_len: {min_len}")
            # # pp
            # ppmin_len = min(len(lst) for lst in pp if len(lst) > 0)
            # # print(f"Rank {rank} rankpp.min_len: {min_len}")
            # # r
            # rmin_len = min(len(lst) for lst in r if len(lst) > 0)
            # # print(f"Rank {rank} rankr.min_len: {min_len}")
            # min_len = min(pmin_len,ppmin_len)
            # min_len = min(min_len,rmin_len)
            # # print(f"Rank {rank} global.min_len: {min_len}")

            # rankp = [torch.tensor(lst[:min_len]) for lst in p]
            # rankp = torch.stack(rankp)  # shape [N, min_len]
            # pad_len = 2500
            # if min_len < pad_len:
            #     pad = torch.zeros(rankp.shape[0], pad_len - min_len)
            #     rankp = torch.cat([rankp, pad], dim=1)
            # if rankp.shape[0] < 13:
            #     pad = torch.zeros(13-rankp.shape[0], pad_len)
            #     rankp = torch.cat([rankp, pad], dim=0)
            # # print(f"Rank {rank} rankp.shape: {rankp.shape}")
            

        
            # rankpp = [torch.tensor(lst[:min_len]) for lst in pp]
            # rankpp = torch.stack(rankpp)  # shape [N, min_len]
            # pad_len = 2500
            # if min_len < pad_len:
            #     pad = torch.zeros(rankpp.shape[0], pad_len - min_len)
            #     rankpp = torch.cat([rankpp, pad], dim=1)
            # if rankpp.shape[0] < 13:
            #     pad = torch.zeros(13-rankpp.shape[0], pad_len)
            #     rankpp = torch.cat([rankpp, pad], dim=0)
            # # print(f"Rank {rank} rankpp.shape: {rankpp.shape}")

        
            # rankr = [torch.tensor(lst[:min_len]) for lst in r]
            # rankr = torch.stack(rankr)  # shape [N, min_len]
            # pad_len = 2500
            # if min_len < pad_len:
            #     pad = torch.zeros(rankr.shape[0], pad_len - min_len)
            #     rankr = torch.cat([rankr, pad], dim=1)
            # if rankr.shape[0] < 13:
            #     pad = torch.zeros(13-rankr.shape[0], pad_len)
            #     rankr = torch.cat([rankr, pad], dim=0)
            # print(f"Rank {rank} rankr.shape: {rankr.shape}")
        
            all_gathered_acceptance = [torch.zeros_like(rank_acceptance) for _ in range(2)]
            torch.distributed.all_gather(all_gathered_acceptance, rank_acceptance)

            # all_gathered_overhead = [torch.zeros_like(rank_oh) for _ in range(8)]
            # torch.distributed.all_gather(all_gathered_overhead, rank_oh)

            # all_gathered_p = [torch.zeros_like(rankp) for _ in range(8)]
            # torch.distributed.all_gather(all_gathered_p, rankp)

            # all_gathered_pp = [torch.zeros_like(rankpp) for _ in range(8)]
            # torch.distributed.all_gather(all_gathered_pp, rankpp)

            # all_gathered_r = [torch.zeros_like(rankr) for _ in range(8)]
            # torch.distributed.all_gather(all_gathered_r, rankr)
        
            def remove_padding(x: torch.Tensor, pad_val: int = 0):
                result = []
                for row in x:
                    # Find where padding starts (assuming trailing padding)
                    non_pad = (row != pad_val).nonzero(as_tuple=True)[0]
                    if len(non_pad) > 0:
                        last_idx = non_pad[-1].item() + 1
                        result.append(row[:last_idx])
                    else:
                        result.append(torch.tensor([], dtype=row.dtype, device=row.device))
                return result
        
            if rank == 0 :
                global_acceptance = torch.cat(all_gathered_acceptance)[:total_prompt_count]        
                global_acceptance = remove_padding(global_acceptance) 
                min_len = min(len(lst) for lst in global_acceptance)
                print(f"Min_len for global accept: {min_len}")
                global_acceptance = [lst[:min_len] for lst in global_acceptance]
                global_acceptance = torch.stack(global_acceptance).cpu().numpy()
                # print(f"Rank {rank} global_acceptance.shape: {global_acceptance.shape}")

                global_overhead = torch.cat(all_gathered_overhead)[:total_prompt_count]        
                global_overhead = remove_padding(global_overhead) 
                min_len = min(len(lst) for lst in global_overhead)
                global_overhead = [lst[:min_len] for lst in global_overhead]
                global_overhead = torch.stack(global_overhead).cpu().numpy()
                avg_overhead = np.mean(global_overhead, axis=0)

                plt.scatter(range(len(avg_overhead)), avg_overhead, marker='o', label=f"Avg Overhead over {total_prompt_count} Images={np.mean(avg_overhead):.2f}")
                plt.xlabel("Generation Index")
                plt.ylabel("Latency (sec)")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"{args.output_dir}/avg-overhead.png")
                plt.close()

                avg_acceptance = np.mean(global_acceptance, axis=0)

                plt.scatter(range(len(avg_acceptance)), avg_acceptance, marker='o', label=f"Avg Acceptance Length over {total_prompt_count} Images={np.mean(avg_acceptance):.2f}")
                plt.xlabel("Generation Index")
                plt.ylabel("Number of Tokens Accepted")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"{args.output_dir}/avg-accept-length.png")
                plt.close()

                # p
                global_p = torch.cat(all_gathered_p)[:total_prompt_count, :1000]             
                # print(f"Rank {rank} global_p.shape: {global_p.shape}")
                # pp
                global_pp = torch.cat(all_gathered_pp)[:total_prompt_count, :1000]             
            
                # print(f"Rank {rank} global_pp.shape: {global_pp.shape}")
                # r
                global_r = torch.cat(all_gathered_r)[:total_prompt_count , :1000]        
                # print(f"Rank {rank} global_r.shape: {global_r.shape}")

                avg_p = np.mean(global_p.cpu().numpy(), axis=0)
                avg_pp = np.mean(global_pp.cpu().numpy(), axis=0)
                avg_r = np.mean(global_r.cpu().numpy(), axis=0)

                plt.scatter(range(len(avg_p)), avg_p, label="Original Confidence Score",  marker='o', alpha=0.5)
                plt.scatter(range(len(avg_pp)), avg_pp, label="LANTERN Confidence Score",  marker='s', alpha=0.5)
                plt.scatter(range(len(avg_r)), avg_r, label="Confidence Threshold", marker='^', alpha=0.5)
                plt.xlabel("Generation Index")
                plt.ylabel("Confidence Scores")
                plt.title("Avg Confidence Scores over 100 Images")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                plt.savefig(f"{args.output_dir}/avg-confidence-scores.png")
                plt.close()

                max_accpt = np.max(global_acceptance, axis=0)
                plt.plot(range(len(max_accpt)), max_accpt, label=f"Max Acceptance Length over 100 Images")
                plt.xlabel("Generation Index")
                plt.ylabel("Number of Tokens Accepted")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"{args.output_dir}/max-avg-accept-length.png")
                plt.close()

        torch.distributed.destroy_process_group()
    else:
        latencies = torch.tensor(latencies)
        latencies = np.array(latencies.cpu())
        avg_latency = np.mean(latencies, axis=0)
        print(f"Avg latency: {avg_latency} per image, with {latencies.shape} images.")

        plt.scatter(range(len(latencies)), latencies, marker='o', label=f"Latency of {total_prompt_count} Images with avg={avg_latency:.2f}")
        plt.xlabel("Generation Index")
        plt.ylabel("Time (Sec)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/5imgs-avg-latency.png")
        plt.close()


        if args.test:
            min_len = min(len(lst) for lst in acceptance_lists)
            acceptance_lists = [torch.tensor(lst[:min_len]) for lst in acceptance_lists]
            acceptance_lists = np.array(acceptance_lists)
            max_accpt = np.max(acceptance_lists, axis=0)
            avg_accpt = np.mean(acceptance_lists, axis=0)

            plt.plot(range(len(max_accpt)), max_accpt, label=f"Max Acceptance Length over 5 Images")
            plt.scatter(range(len(avg_accpt)), avg_accpt, marker='o', label=f"Avg Acceptance Length over 5 Images={np.mean(avg_accpt):.2f}")
            plt.xlabel("Generation Index")
            plt.ylabel("Number of Tokens Accepted")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{args.output_dir}/5imgs-avg-accept-length.png")
            plt.close()

          
            # min_len = min(len(lst) for lst in overheads)
            # l_overhead = [lst[:min_len] for lst in overheads]
            # l_overhead = torch.stack(overheads).cpu().numpy()
            # avg_overhead = np.mean(l_overhead, axis=0)

            # plt.scatter(range(len(l_overhead)), l_overhead, marker='o', label=f"Avg Overhead over 100 Images={np.mean(avg_overhead):.2f}")
            # plt.xlabel("Generation Index")
            # plt.ylabel("Latency (sec)")
            # plt.legend()
            # plt.grid(True)
            # plt.tight_layout()
            # plt.savefig(f"{args.output_dir}/5imgs-avg-overhead.png")
            # plt.close()


            

    with open(f"{args.output_dir}/global_statistics_{rank}_{start_idx}_{end_idx}.json", "w") as f:
        json.dump(global_statistics, f, indent=4)

    with open(f"{args.output_dir}/generation_configs_{rank}.json", "w") as f:
        json.dump(vars(args), f, indent=4)

if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

=======
import re
import os
import json, csv
import random
import argparse
import traceback

import torch
import numpy as np

import models.drafters.choices as choices

from itertools import product
from torchvision.utils import save_image

from tqdm import tqdm

USE_EXPERIMENTAL_FEATURES = os.getenv("USE_EXPERIMENTAL_FEATURES", "0") == "1"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model to use for image generation",
                        default="lumina_mgpt")
    parser.add_argument("--model_type", type=str, help="Model type; choices: ['eagle', 'base', 'vllm']",
                        default="eagle")
    parser.add_argument("--model_path", type=str, help="Path to the model",
                        default="Alpha-VLLM/Lumina-mGPT-7B-768")
    parser.add_argument("--drafter_path", type=str, help="Path to the drafter model",
                        default="ckpts/lumina_mgpt/trained_drafters/lumina_mgpt_7b_768_lr0.0001_p_w0.1"
                        "_bsz2_gradacc_8_epochs20_noise_uniform_std0.2_coupled_False_cfgloss_False"
                        "_cfgscale_3.0_embed_upscale1.0_mscoco2017train30k/state_20")
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--target_size", type=int, default=768)

    parser.add_argument("--prompt", type=str, help="Prompt for image generation",
                        default="PartiPrompts")
    parser.add_argument("--num_images", type=int, help="Number of images to generate",
                        default=10)
    parser.add_argument("--slice", type=str, help="Slice of prompts to use; format: 'start-end'",
                        default=None)
    parser.add_argument("--output_dir", type=str, help="Output directory for generated images",
                        default="generated_images")
    
    parser.add_argument("--set_seed", action="store_true", help="Set random seed for reproducibility")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation")
    parser.add_argument("--top_k", type=int, default=2000, help="Top-k for generation")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for generation")
    parser.add_argument("--cfg", type=float, default=3.0, help="CFG for generation")
    
    # LANTERN-specific arguments
    parser.add_argument("--lantern", action="store_true", help="Use LANTERN for image generation")
    parser.add_argument("--lantern_k", type=int, default="1000", help="Value of k for LANTERN")
    parser.add_argument("--lantern_delta", type=float, default="0.1", help="Value of delta for LANTERN")
    parser.add_argument("--grid_search", action="store_true", help="Run grid search for LANTERN hyperparameters")

    parser.add_argument("--static_tree", action="store_true", help="Use static tree based drafting")
    parser.add_argument("--eagle_version", type=int, default=1, help="EAGLE version")
    parser.add_argument("--cfg_mode", type=str, default="sequential", help="CFG mode")

    # Experimental arguments
    parser.add_argument("--tree_choices", type=str, help="Tree choice for LANTERN",
                        default="mc_sim_7b_63")
    parser.add_argument("--drafter_top_k", type=int, default=None, help="Top-k for drafter")

    # legacy arguments
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for image generation")
    parser.add_argument("--end_idx", type=int, default=10000, help="End index for image generation")

    return parser

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model(args):
    if args.model == "lumina_mgpt":
        if args.model_type == "vllm":
            from models.base_models.lumina_mgpt.vllm_inference_solver import FlexARInferenceSolver
            model = FlexARInferenceSolver(
                model_path=args.model_path,
                precision=args.precision,
                target_size=args.target_size,
                max_num_seqs=24,
            )

        elif args.model_type == "base":
            from models.base_models.lumina_mgpt.inference_solver import FlexARInferenceSolver
            model = FlexARInferenceSolver(
                model_path=args.model_path,
                precision=args.precision,
                target_size=args.target_size,
                cfg_mode=args.cfg_mode,
            )

        elif args.model_type == "eagle":
            from models.base_models.lumina_mgpt.eagle_inference_solver import FlexARInferenceSolver
            model = FlexARInferenceSolver(
                model_path=args.model_path,
                drafter_path=args.drafter_path,
                precision=args.precision,
                target_size=args.target_size,
                cfg_mode=args.cfg_mode,
                eagle_version=args.eagle_version,
            )
        else:
            raise ValueError(f"Model type {args.model_type} is not supported for model {args.model}")
    elif args.model == "anole":
        if args.model_type == "vllm":
            raise NotImplementedError("VLLM model is not supported for Anole model")
        
        elif args.model_type == 'base':
            from models.kv_variants.modeling_anole_kv import ChameleonForConditionalGeneration
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
            model = ChameleonForConditionalGeneration.from_pretrained(args.model_path).to(dtype=dtype, device='cuda')
            model.eval()
        
        elif args.model_type == 'eagle':
            from models.ea_model_anole import EaModel
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
            model = EaModel.from_pretrained(base_model_path=args.model_path, ea_model_path=args.drafter_path).to(dtype=dtype, device='cuda')
            model.eval()
        
        else:
            raise ValueError(f"Model type {args.model_type} is not supported for model {args.model}")
    elif "llamagen" in args.model:
        if args.model_type == "vllm":
            raise NotImplementedError("VLLM model is not supported for LLAMAGen model")
        
        elif args.model_type == 'base':
            from models.kv_variants.modeling_llamagen_kv import LlamaForCausalLM
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
            model = LlamaForCausalLM.from_pretrained(args.model_path).to(dtype=dtype, device='cuda')
            model.eval()
        
        elif args.model_type == 'eagle':
            from models.ea_model_llamagen import EaModel
            dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
            model = EaModel.from_pretrained(base_model_path=args.model_path, ea_model_path=args.drafter_path).to(dtype=dtype, device='cuda')
            model.eval()
        else:
            raise ValueError(f"Model type {args.model_type} is not supported for model {args.model}")
    else:
        raise NotImplementedError(f"Model {args.model} is not supported")
    
    return model

def load_prompts(args):
    prompts = []
    if args.prompt == "PartiPrompts":
        with open('data/prompts/PartiPrompts.tsv', 'r') as f:
            tsv_reader = csv.DictReader(f, delimiter='\t')
            for row in tsv_reader:
                prompts.append(row['Prompt'])
    elif args.prompt == "MSCOCO2017Val":
        with open('data/prompts/captions_val2017_longest.json', 'r') as f:
            captions = json.load(f)
            for caption in captions:
                prompts.append(caption)
    elif args.prompt == "MSCOCO2014Val":
        with open('data/prompts/captions_val_2014.json', 'r') as f:
            captions = json.load(f)
            for caption in captions:
                prompts.append(caption)
    elif args.prompt == "MSCOCO2017Train":
        with open('data/prompts/captions_train2017_extracted.json', 'r') as f:
            captions = json.load(f)
            for caption in captions:
                prompts.append(caption['caption'])
    elif args.prompt == "SJDPrompts":
        with open('data/prompts/SJDPrompts.tsv', 'r') as f:
            tsv_reader = csv.DictReader(f, delimiter='\t')
            for row in tsv_reader:
                prompts.append(row['Prompt'])
    else:
        # Single prompt input
        prompts = [args.prompt] * args.num_images

    if args.slice is not None:
        assert re.match(r'^\d+-\d+$', args.slice), f"Invalid format: '{args.slice}'. Expected format is 'start-end'."

        start, end = map(int, args.slice.split('-'))
        assert start < end, f"Invalid range: '{args.slice}'. Start value must be less than end value."
        assert start >= 0 and end >= 0, "Slice values must be non-negative."

        prompts = prompts[start:end]
    
    if args.num_images < len(prompts):
        print(f"Number of images to generate is less than the number of prompts. Sampling {args.num_images} prompts.")
        prompts = random.sample(prompts, args.num_images)
    else:
        print(f"Number of images to generate is greater than the number of prompts. Generating only {len(prompts)} images and no sampling.")
        pass
    
    return prompts

def generate_and_save_image(model, model_name, prompt, img_save_path, **kwargs):
    # print(f"Generating image for prompt: {prompt}")
    if model_name == "lumina_mgpt":
        generate_params = {
            "images": [],
            "qas": [[prompt, None]],
            "max_gen_len": 2354,
            "temperature": kwargs["temperature"],
            "top_k": kwargs["top_k"],
            "cfg_scale": kwargs["cfg"],
        }
    elif model_name in ["anole", "llamagen", "llamagen2"]:
        if model_name == "llamagen":
            max_gen_len = 256
        else:
            max_gen_len = 1024
        generate_params = {
            "prompt": [prompt],
            "max_length": max_gen_len,
            "temperature": kwargs["temperature"],
            "top_k": kwargs["top_k"],
            "top_p": kwargs["top_p"],
            "cfg": kwargs["cfg"],
            "static_tree": kwargs["static_tree"],
        }
    else:
        raise NotImplementedError(f"Model {model_name} is not supported for image generation.")
    
    if "lantern" in kwargs:
        generate_params["lantern"] = kwargs.get("lantern", False)
        generate_params["lantern_k"] = kwargs.get("lantern_k", 1000)
        generate_params["lantern_delta"] = kwargs.get("lantern_delta", 0.1)

    if USE_EXPERIMENTAL_FEATURES:
        generate_params["tree_choices"] = kwargs["tree_choices"]
        generate_params["drafter_top_k"] = kwargs["drafter_top_k"]

    generated_tokens, step_compression, latency = model.generate(**generate_params)
    _, generated_image = model.decode_ids(generated_tokens)

    if model_name in ["lumina_mgpt", "anole"]:
        generated_image[0].save(img_save_path, "png")
    elif "llamagen" in model_name:
        save_image(generated_image, img_save_path, normalize=True, value_range=(-1, 1))

    return step_compression, latency

def run_generate_image(args):
    assert args.model_type != "vllm", "VLLM model is not supported for single image generation"

    if args.set_seed:
        set_seed(args.random_seed)
    
    model = load_model(args)
    prompts = load_prompts(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    global_statistics = {}  
    for idx, prompt in tqdm(enumerate(prompts), total=len(prompts)):
        if idx < args.start_idx or idx >= args.end_idx:
            continue
        if args.model == "lumina_mgpt":
            q1 = f"Generate an image of 768x768 according to the following prompt:\n{prompt}"
        else:
            q1 = prompt

        generate_image_kwargs = {
            "model" : model,
            "model_name" : args.model,
            "prompt" : q1,
            "temperature" : args.temperature,
            "top_k" : args.top_k,
            "top_p" : args.top_p,
            "cfg" : args.cfg,
            "lantern": args.lantern,
            "lantern_k": args.lantern_k,
            "lantern_delta": args.lantern_delta,
            "img_save_path": f"{args.output_dir}/prompt_{idx}.png",
            "static_tree": args.static_tree,
        }

        if USE_EXPERIMENTAL_FEATURES:
            try:
                tree_choices = getattr(choices, args.tree_choices)
            except AttributeError:
                print(f"Tree choices {args.tree_choices} is not a valid choice")
                return
            
            generate_image_kwargs["tree_choices"] = tree_choices
            generate_image_kwargs["drafter_top_k"] = args.drafter_top_k
        
        step_compression, latency = generate_and_save_image(**generate_image_kwargs)

        statistics = {
            "prompt": prompt,
            "step_compression": step_compression,
            "latency": latency
        }

        global_statistics[f"prompt_{idx}"] = statistics

    with open(f"{args.output_dir}/global_statistics_{args.start_idx}_{args.end_idx}.json", "w") as f:
        json.dump(global_statistics, f, indent=4)

    with open(f"{args.output_dir}/generation_configs.json", "w") as f:
        json.dump(vars(args), f, indent=4)

if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

>>>>>>> eagle3
    run_generate_image(args)