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
from models.base_models.januspro.modeling_vlm import (
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
    # def __init__(self, drafter_config, head, lm_head, embed_tokens, norm, rotary_emb, top_k, total_tokens, depth, threshold, target_hidden_size=4096):
    def __init__(self, drafter_config, top_k, total_tokens, depth, threshold, target_hidden_size=4096):
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
        
        self.top_k = top_k
        self.total_tokens = total_tokens - 1
        self.depth = depth
        self.threshold = math.log(threshold)

        # 3. Setup EAGLE specific components
        # Use target_hidden_size * 2 because we concat (H_t and E_{t+1})
        self.drafter_h = drafter_config.language_config.hidden_size
        self.fusion = torch.nn.Linear(target_hidden_size * 2, self.drafter_h)
        self.fusion_act = torch.nn.SiLU()

        # # 4. Attach the base model head
        # self.gen_head = copy.deepcopy(head)  
        # self.language_model.lm_head = copy.deepcopy(lm_head) 
        # self.language_model.model.embed_tokens = copy.deepcopy(embed_tokens)
        # self.language_model.model.norm = copy.deepcopy(norm)  
        # self.language_model.model.rotary_emb = copy.deepcopy(rotary_emb)   
        
        # for param in self.gen_head.parameters():
        #     param.requires_grad = False
        # self.gen_head.eval()
        
        # for param in self.language_model.lm_head.parameters():
        #     param.requires_grad = False
        # self.language_model.lm_head.eval()
        
        # for param in self.language_model.model.embed_tokens.parameters():
        #     param.requires_grad = False
        # self.language_model.model.embed_tokens.eval()
        
        # for param in self.language_model.model.rotary_emb.parameters():
        #     param.requires_grad = False
        # self.language_model.model.rotary_emb.eval()
        
        # for param in self.language_model.model.norm.parameters():
        #     param.requires_grad = False
        # self.language_model.model.norm.eval()
        

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

    def forward(self, hidden_states, input_embeds, past_key_values=None, position_ids=None, use_cache=True, **kwargs) :
        """Standard Forward Pass for Training/Inference."""
        # 1. Get embeddings for the tokens using the inherited Janus method
        # This uses the Drafter's (lightweight) embedding layer
        # input_embeds = self.prepare_gen_img_embeds(input_ids)
        
        # 2. EAGLE Fusion
        combined = torch.cat([hidden_states, input_embeds], dim=-1)
        fused_states = self.fusion_act(self.fusion(combined))
        
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = hidden_states.device if hidden_states is not None else input_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()


        # 3. Pass through the 1-layer Transformer
        # We call the language_model (Llama) inherited from MultiModalityCausalLM
        
        outputs = self.language_model.model(
            inputs_embeds=fused_states,
            past_key_values=past_key_values, 
            position_ids=position_ids,
            use_cache=use_cache,
            **kwargs
        )
        
        return outputs.last_hidden_state, outputs.past_key_values
    
    def sample(self, logits, k, temperature):
       
        probabilities = torch.nn.functional.softmax(logits/ temperature, dim=1)
        
        bias = []
        normalized = F.normalize(logits.clone().to(torch.float32), dim=1, eps=1e-6)
        if normalized.shape[0] > 1 :
            # Compute cosine similarity matrix: [B, B]
            cosine_sim_matrix = torch.matmul(normalized, normalized.T)
            # Keep scores where similarity > threshold
            high_sim_mask = cosine_sim_matrix > 0.9  # shape [B, B]
            # Get indices
            rows, cols = torch.nonzero(high_sim_mask, as_tuple=True)
            for r,c in zip(rows.tolist(), cols.tolist()):
                if r != c:
                    bias.append((r,c))
                    

        sampled_indices = torch.multinomial(probabilities, k, replacement=False)
        sampled_probs = torch.gather(probabilities, 1, sampled_indices)

        cumulative_sum = torch.cumsum(sampled_probs, dim=1)
        cumulative_sum = torch.cat(
            (torch.zeros(cumulative_sum.shape[0], 1, device=cumulative_sum.device), cumulative_sum[:, :-1]), dim=-1)

        sampled_probs = sampled_probs / (1 - cumulative_sum)
        sampled_probs[torch.isinf(sampled_probs)] = -1
        sampled_probs[torch.isnan(sampled_probs)] = -1

        sampled_probs = torch.clamp(sampled_probs, min=0.0, max=1.0)
        return sampled_indices, sampled_probs,probabilities, bias
    
    def init_tree(self):
        self.tree_mask_init = torch.eye(self.top_k, device=self.device)[None, None]
        self.position_ids = torch.zeros(self.top_k, device=self.device, dtype=torch.long)

    def init_tree_v1(self, tree_choices):
        self.tree = tree_choices
        self.tree_buffer=generate_tree_buffers(self.tree, "cuda")
        
    @torch.no_grad()
    def topK_genrate(
        self,
        hidden_state: torch.Tensor,     # (2, T, D) — hidden states at accepted positions
        image_embeds: torch.Tensor,        # (2, seq_len) — cond + uncond, with last token
        head,                           # gen_head from target (or drafter — same interface)
        cfg_scale: float,
        temperature=1.0,
    ) -> Tuple[tuple, list, list]:
        """Generate draft token tree.

        input_ids is (2, T): row 0 = cond, row 1 = uncond (same as LlamaGen's
        batch-duplicated input). The drafter runs both in one pass and splits
        logits with cfg_logit_process.
        """
        image_embeds = image_embeds.to(hidden_state.device)
        # image_embeds = image_embeds[:, 1:, :]
        ss_token = []
        ss_prob  = []
        ss_op    = []

        # Initial drafter hidden state = target's accepted hidden state
        len_posi   = image_embeds.shape[1] 
        image_embeds = image_embeds[:, -1:, :] # always see the very last token 
        
        # past_key_values = (key_value,) for only one layer
        out_hidden, past_key_values = self(
            hidden_state, 
            image_embeds,
            use_cache  = True,
        )
        last_hidden = out_hidden[:, -1]
        last_hidden = head(last_hidden)
        last_headout = cfg_logit_process(last_hidden, cfg_scale)   
        
        bias_list = [ [] for x in range(len(self.tree_buffer['tree_indices'])+1)]
        
        # last_headout = mask_non_image_logits(last_headout)
        for i in range(len(self.tree_buffer["tree_indices"])):
            topk_index,topk_prob,op, bias  = self.sample(last_headout, 10, temperature)
            bias_list[i] = bias

            ss_token.append(topk_index)
            ss_prob.append(topk_prob)
            ss_op.append(op)

            # Build input_ids for this level — select tree nodes
            topk_flat  = topk_index.view(-1)
            select_index = topk_flat[self.tree_buffer["tree_indices"][i]]   # (n_nodes,)

            input_ids = select_index[None,:]
            input_ids = torch.cat([input_ids, input_ids])
            inputs_embeds    = self.prepare_gen_img_embeds(input_ids)

            # Repeat hidden states to match tree branching
            if i == 0:
                hidden_states = out_hidden[:, -1:]
            else:
                hidden_states = out_hidden
            hidden_states = self.repeat_hidden(hidden_states, self.tree_buffer["repeat_nums"][i])

            # Set tree attention mask
            self.tree_mask = self.tree_buffer["attn_mask"][i]
            position_ids_step = len_posi + self.tree_buffer["position_ids"][i]

            # Drafter forward — inputs_embeds from prepare_gen_img_embeds
            out_hidden, past_key_values = self(
                hidden_states, 
                inputs_embeds,
                past_key_values, 
                position_ids_step,
                use_cache       = True,
            )
            len_posi        += 1

            # gen_head on drafter output, then CFG
            last_headout   = head(out_hidden)
            last_headout  = cfg_logit_process(last_headout, cfg_scale).squeeze(0)  # (n_nodes, V)

        # Final level
        topk_index,topk_prob,op, bias  = self.sample(last_headout, 10, temperature)

        ss_token.append(topk_index)
        ss_prob.append(topk_prob)
        ss_op.append(op)
        bias_list[len(self.tree_buffer['tree_indices'])] = bias
       
        return (torch.cat(ss_token), torch.cat(ss_prob), ss_op), bias_list

