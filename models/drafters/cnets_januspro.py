import copy
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import math
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import json
from .utils_c import *
from transformers import AutoModelForCausalLM, AutoConfig
from janus.models.modeling_vlm import (
    MultiModalityCausalLM, 
    VisionConfig, AlignerConfig, GenVisionConfig, 
    GenAlignerConfig, GenHeadConfig
)
from transformers import LlamaConfig

from models.configs.configs import EConfig
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
from .choices import *
from safetensors import safe_open
# ---------------------------------------------------------------------------
# Drafter: Janus-Pro-1B running the same gen_head pipeline
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
from typing import Tuple, List, Optional
from transformers.models.llama.modeling_llama import LlamaDecoderLayer # The 7B-compatible layer
from janus.models.modeling_vlm import MultiModalityCausalLM

def cfg_logit_process(
    combined_logits: torch.Tensor,
    cfg_scale: float = 5.0,
) -> torch.Tensor:
    """CFG for Janus-Pro — even rows are cond, odd rows are uncond.
 
    Matches the official reference:
        logit_cond   = logits[0::2]
        logit_uncond = logits[1::2]
        out = logit_uncond + cfg_scale * (logit_cond - logit_uncond)
    """
    logit_cond   = combined_logits[0::2]
    logit_uncond = combined_logits[1::2]
    return logit_uncond + cfg_scale * (logit_cond - logit_uncond)

class Model(MultiModalityCausalLM):
    def __init__(self, drafter_config, head, target_hidden_size=4096):
        # 1. Ensure all sub-configs are the correct Object Types
        # This prevents the "'dict' object has no attribute 'cls'" error
        
        sub_configs = {
            "vision_config": VisionConfig,
            "aligner_config": AlignerConfig,
            "gen_vision_config": GenVisionConfig,
            "gen_aligner_config": GenAlignerConfig,
            "gen_head_config": GenHeadConfig,
        }

        for attr, config_cls in sub_configs.items():
            current_val = getattr(drafter_config, attr)
            if isinstance(current_val, dict):
                setattr(drafter_config, attr, config_cls(**current_val))
        
        # Handle language_config separately (usually LlamaConfig)
        if isinstance(drafter_config.language_config, dict):
            drafter_config.language_config = LlamaConfig(**drafter_config.language_config)

        # 2. Now call super() - Janus will now find the .cls and .params attributes
        super().__init__(drafter_config)

        # 3. Setup EAGLE specific components
        # Use target_hidden_size * 2 because we concat (H_t and E_{t+1})
        self.drafter_h = drafter_config.language_config.hidden_size
        self.fusion = torch.nn.Linear(target_hidden_size * 2, self.drafter_h)
        self.fusion_act = torch.nn.SiLU()

        # 4. Attach the base model head
        self.gen_head = head
        for param in head.parameters():
            param.requires_grad = False

    def repeat_hidden(self, hidden: torch.Tensor, repeat_nums: List[int]) -> torch.Tensor:
        """Splits and repeats Cond/Uncond streams for Janus CFG branching."""
        # hidden: [2, N, D] (0: cond, 1: uncond)
        cond_h, uncond_h = hidden[0:1], hidden[1:2]
        
        new_cond = []
        new_uncond = []
        for i, n in enumerate(repeat_nums):
            new_cond.append(cond_h[:, i:i+1, :].expand(-1, n, -1))
            new_uncond.append(uncond_h[:, i:i+1, :].expand(-1, n, -1))
            
        return torch.cat([torch.cat(new_cond, dim=1), torch.cat(new_uncond, dim=1)], dim=0)

    def forward(self, hidden_states, input_embeds, **kwargs) :
        """Standard Forward Pass for Training/Inference."""
        # 1. Get embeddings for the tokens using the inherited Janus method
        # This uses the Drafter's (lightweight) embedding layer
        # input_embeds = self.prepare_gen_img_embeds(input_ids)
        
        # 2. EAGLE Fusion
        combined = torch.cat([hidden_states, input_embeds], dim=-1)
        fused_states = self.fusion_act(self.fusion(combined))
        
        # 3. Pass through the 1-layer Transformer
        # We call the language_model (Llama) inherited from MultiModalityCausalLM
        outputs = self.language_model.model(
            inputs_embeds=fused_states,
            **kwargs
        )
        
        return outputs.last_hidden_state
    def sample(self, logits, k=1):
       
        probabilities = torch.nn.functional.softmax(logits, dim=1)

        sampled_indices = torch.multinomial(probabilities, k, replacement=False)
        sampled_probs = torch.gather(probabilities, 1, sampled_indices)

        cumulative_sum = torch.cumsum(sampled_probs, dim=1)
        cumulative_sum = torch.cat(
            (torch.zeros(cumulative_sum.shape[0], 1, device=cumulative_sum.device), cumulative_sum[:, :-1]), dim=-1)

        sampled_probs = sampled_probs / (1 - cumulative_sum)
        sampled_probs[torch.isinf(sampled_probs)] = -1
        sampled_probs[torch.isnan(sampled_probs)] = -1

        sampled_probs = torch.clamp(sampled_probs, min=0.0, max=1.0)
        return sampled_indices, sampled_probs,probabilities
    
    @torch.no_grad()
    def topK_genrate(
        self,
        hidden_state: torch.Tensor,     # (2, T, D) — hidden states at accepted positions
        input_ids: torch.Tensor,        # (2, seq_len) — cond + uncond, with last token
        head,                           # gen_head from target (or drafter — same interface)
        cfg_scale: float,
        past_key_values=None,
    ) -> Tuple[tuple, list, list]:
        """Generate draft token tree.

        input_ids is (2, T): row 0 = cond, row 1 = uncond (same as LlamaGen's
        batch-duplicated input). The drafter runs both in one pass and splits
        logits with cfg_logit_process.
        """
        device   = input_ids.device
        ss_token = []
        ss_prob  = []
        ss_op    = []

        # Initial drafter hidden state = target's accepted hidden state
        out_hidden = hidden_state          # (2, T, D)
        len_posi   = input_ids.shape[1] - 1

        # First gen_head call on target hidden state
        last_headout = head(out_hidden[:, -1:, :]).squeeze(1)   # (2, 1, V)
        last_headout = cfg_logit_process(last_headout, cfg_scale)   
        print("last_headout shape: ", last_headout.shape, flush=True)# (1, V) after CFG
        # last_headout = mask_non_image_logits(last_headout)
        for x in range(len(self.tree_buffer["tree_indices"])):
            print(f"self.tree_buffer['tree_indices'][{x}]: ", self.tree_buffer["tree_indices"][x], flush=True)
        for i in range(len(self.tree_buffer["tree_indices"])):
            topk_index,topk_prob,op  = self.sample(last_headout, 10)

            ss_token.append(topk_index)
            ss_prob.append(topk_prob)
            ss_op.append(op)

            # Build input_ids for this level — select tree nodes
            topk_flat  = topk_index.view(-1)
            print("topk_flat shape: ", topk_flat.shape, flush=True)
            select_ids = topk_flat[self.tree_buffer["tree_indices"][i]]   # (n_nodes,)
            print("select_ids shape: ", select_ids.shape, flush=True)

            # Embed using prepare_gen_img_embeds — same as official inference
            # Duplicate for cond + uncond (same token, CFG handled by interleaving)
            select_ids_cfg = select_ids.unsqueeze(0).repeat(2,1)   # (2*n_nodes,)
            print("select_ids_cfg shape: ", select_ids_cfg.shape, flush=True)
            node_embeds    = self.prepare_gen_img_embeds(select_ids_cfg)
            print("node_embeds shape: ", node_embeds.shape, flush=True)
            # # (2*n_nodes, D) — reshape to (2, n_nodes, D) then (1, n_nodes, D) per pass
            # # For drafter single batch: use (1, n_nodes, D) shape
            # node_embeds_cond = node_embeds[0::2].unsqueeze(0)   # (1, n_nodes, D)

            # Repeat hidden states to match tree branching
            if i == 0:
                h_step = out_hidden[:, -1:]
            else:
                h_step = out_hidden
            print("h_step shape: ", h_step.shape, flush=True)
            print("self.tree_buffer['repeat_nums'][i]: ", self.tree_buffer["repeat_nums"][i], flush=True)
            h_step = self.repeat_hidden(h_step, self.tree_buffer["repeat_nums"][i])
            print("repeated h_step shape: ", h_step.shape, flush=True)

            # Set tree attention mask
            self.tree_mask = self.tree_buffer["attn_mask"][i]
            position_ids_step = len_posi + self.tree_buffer["position_ids"][i]

            # Drafter forward — inputs_embeds from prepare_gen_img_embeds
            drafter_out = self.language_model.model(
                inputs_embeds   = node_embeds,
                past_key_values = past_key_values,
                position_ids    = position_ids_step.unsqueeze(0),
                use_cache       = True,
            )
            out_hidden       = drafter_out.last_hidden_state   
            print("repeated out_hidden shape: ", out_hidden.shape, flush=True)
            past_key_values  = drafter_out.past_key_values
            len_posi        += 1

            # gen_head on drafter output, then CFG
            raw_headout   = head(out_hidden)
            last_headout  = cfg_logit_process(raw_headout, cfg_scale)  # (n_nodes, V)
            # last_headout  = mask_non_image_logits(last_headout)

        # Final level
        topk_index,topk_prob,op  = self.sample(last_headout, 10)

        ss_token.append(topk_index)
        ss_prob.append(topk_prob)
        ss_op.append(op)
       
        return (torch.cat(ss_token), torch.cat(ss_prob), ss_op)

