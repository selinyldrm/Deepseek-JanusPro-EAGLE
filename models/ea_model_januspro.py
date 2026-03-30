"""
EAGLE-based Speculative Decoding for Janus-Pro
================================================
Adapted from ea2_model_llamagen.py + cnets2_llamagen.py

CFG implementation (from official Janus-Pro inference code):
    - Batch is (parallel_size*2, seq_len)
    - Even rows  [0::2] = conditional
    - Odd  rows  [1::2] = unconditional (prompt tokens 1:-1 replaced with pad_id)
    - Single forward pass through language_model.model
    - gen_head on last hidden state
    - logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
    - Next token embedding via mmgpt.prepare_gen_img_embeds(next_token)

This is identical to LlamaGen's batch-duplication CFG, just with:
    - gen_head instead of lm_head
    - prepare_gen_img_embeds instead of tok_embeddings for image tokens
    - get_input_embeddings() for the prompt prefix (not T5 prefix tokens)
"""

from __future__ import annotations

import copy
import random
import time
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformers import AutoModelForCausalLM, AutoConfig

# from .base_models.januspro.image_processing_vlm import VLMImageProcessor
from models.base_models.januspro.modeling_vlm import MultiModalityCausalLM # [SY]: This imports models/kv_variants/januspro/LlamaForCausalLM
from models.base_models.januspro.processing_vlm import VLChatProcessor

from models.drafters.choices import naive_extend_57

from safetensors.torch import load_file
import os
from huggingface_hub import hf_hub_download

import copy
import json
import time
from typing import List, Optional

from torchvision.utils import save_image
import math
import matplotlib.pyplot as plt

from PIL import Image

from .drafters.utils import *

from .drafters.cnets_januspro import Model, cfg_logit_process
from .configs.configs import EConfig

from .drafters.choices import *

import torch.nn.functional as F
import copy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOPK = 10                    # default draft top-k, overridden by EaModel.top_k


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class KVCache:
    """
    A key-value cache for the model.

    This class provides a mechanism to maintain a growing cache of keys and values,
    particularly useful for models that benefit from caching previous states,
    like transformers during autoregressive decoding.

    Attributes:
        data (torch.Tensor): The tensor storing keys and values.
        current_length (int): Current length of the data being stored.
    """

    def __init__(self, data, current_length):
        """
        Initialize the KVCache.

        Args:
            data (torch.Tensor): Initial tensor to store the keys and values.
            current_length (int): Initial length of the data.
        """
        self.data = data
        self.current_length = current_length

    @property
    def shape(self):
        """Return the shape of the data tensor with updated length."""
        return (
            self.data.shape[0],
            self.data.shape[1],
            self.current_length.item(),
            self.data.shape[3],
        )
    def get_seq_length(self):
        # Transformers calls this to find where to start appending
        return self.current_length.item()

    def copy(self, indices: torch.Tensor, prev_length: int, dim: int = 2):
        """
        Copy values from the current data at specified indices to a new location.

        Args:
            indices (torch.Tensor): Indices of the data tensor to be copied.
            prev_length (int): Previous length before adding new data.
            dim (int, optional): Dimension along which copying should be performed. Default is 2.
        """
        tgt = self.data.index_select(dim, indices)
        dst = self.data.narrow(dim, prev_length, tgt.shape[dim])
        dst.copy_(tgt, non_blocking=True)
        self.current_length.fill_(prev_length + tgt.shape[dim])

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        """
        Concatenate the given tensor with the current data.

        Args:
            tensor (torch.Tensor): The tensor to be concatenated.
            dim (int, optional): The dimension along which concatenation should be done. Default is 2.

        Returns:
            torch.Tensor: The data tensor after concatenation up to the current length.
        """
        dst = self.data.narrow(dim, self.current_length, tensor.shape[dim])
        dst.copy_(tensor)
        self.current_length.add_(tensor.shape[dim])
        return torch.narrow(self.data, 2, 0, self.current_length)

def januspro_initialize_past_key_values(model: MultiModalityCausalLM):
    """Pre-allocate a static KV cache for Janus-Pro's language_model.

    Janus-Pro uses language_model (DeepSeek-based LLM) which has standard
    (key, value) per-layer KV cache. Batch size = parallel_size * 2 for CFG.
    """
    config       = model.language_model.config
    num_layers   = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim     = config.hidden_size // config.num_attention_heads
    max_length   = 4096
    device       = next(model.language_model.parameters()).device
    dtype        = next(model.language_model.parameters()).dtype
    batch_size   = 2   # cond + uncond; scaled by parallel_size in generate()

    startnum=0
    devices=[]
    for i in range(num_layers):
        try:
            device = model.language_model.model.layers[i].self_attn.q_proj.weight.device
        except:
            device=model.language_model.model.layers[i].self_attn.q_proj.weight.device
        devices.append(device)
    startdevice=devices[0]
    past_key_values_data_list=[]
    for id,i in enumerate(devices):
        if startdevice!=i:
            past_key_values_data = torch.zeros(
                startnum * 2,
                batch_size,
                config.num_key_value_heads,
                config.max_position_embeddings,
                config.hidden_size // config.num_attention_heads,
                device=startdevice,
                dtype=model.dtype,
            )
            past_key_values_data_list.append(past_key_values_data)
            startdevice = i
            startnum=0
        startnum += 1
    past_key_values_data = torch.zeros(
        startnum * 2,
        batch_size,
        config.num_key_value_heads,
        config.max_position_embeddings,
        config.hidden_size // config.num_attention_heads,
        device=startdevice,
        dtype=model.dtype,
    )
    past_key_values_data_list.append(past_key_values_data)
    past_key_values = [] * config.num_hidden_layers
    current_length_data = torch.zeros(
        num_layers * 2, dtype=torch.long, device="cpu"
    )

    bias=0
    start_data_m=devices[0].index
    for i in range(config.num_hidden_layers):
        data_m=devices[i].index
        if data_m!=start_data_m:
            bias=0
            start_data_m=data_m
        try:
            past_key_values.append(
                [
                    KVCache(past_key_values_data_list[data_m-devices[0].index][2*bias + j], current_length_data[i * 2 + j])
                    for j in range(2)
                ]
            )
        except:
            past_key_values.append(
                [
                    KVCache(past_key_values_data_list[0][2 * bias + j],
                            current_length_data[i * 2 + j])
                    for j in range(2)
                ]
            )
        bias+=1

    return past_key_values, past_key_values_data, current_length_data

def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits




def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    sample_logits: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample one token from image logits.

    Args:
        logits: (B, V) — already CFG-combined gen_head output
    Returns:
        idx:   (B, 1)
        probs: (B, V)
    """
    logits = logits / max(temperature, 1e-5)
    # mask_non_image_logits(logits)
    # if top_k > 0 or top_p < 1.0:
    #     logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs


# ---------------------------------------------------------------------------
# Main EaModel for Janus-Pro
# ---------------------------------------------------------------------------

class EaModel(nn.Module):
    """EAGLE speculative decoding for Janus-Pro.

    Mirrors ea2_model_llamagen.py EaModel with Janus-Pro-specific:
    - Single-pass CFG via even/odd batch interleaving
    - prepare_gen_img_embeds for image token embeddings
    - gen_head instead of lm_head
    - Image token range masking [100015, 116399)
    """

    def __init__(
        self,
        base_model: MultiModalityCausalLM,
        drafter_config: AutoConfig,
        ea_layer_state_dict: torch.Tensor,
        processor: VLChatProcessor,
        top_k: int   = 10,
        total_tokens: int = 59,
        depth: int   = 5,
        threshold: float = 1.0,
    ):
        super().__init__()
        self.base_model = base_model.eval()
        self.processor  = processor
        self.config     = base_model.config
        self.hidden_size = base_model.language_model.config.hidden_size
        self.vocab_size  = base_model.language_model.config.vocab_size

        self.ea_layer = Model(
            drafter_config,
            base_model.gen_head,
            top_k ,
            total_tokens,
            depth,
            threshold,
        ).eval()
        
        self.ea_layer.load_state_dict(ea_layer_state_dict, strict=True)
        device = "cuda"
        self.ea_layer.to(self.base_model.dtype).to(device)
        self.ea_layer.init_tree()
        self.ea_layer.to(base_model.language_model.dtype).to(device)
        self.device = device
        
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        for param in self.base_model.gen_vision_model.parameters():
            param.requires_grad = False
        for param in self.base_model.language_model.parameters():
            param.requires_grad = False
        for param in self.base_model.gen_embed.parameters():
            param.requires_grad = False
        for param in self.base_model.vision_model.parameters():
            param.requires_grad = False
            
        # Check if gen_head is on a different device (model parallelism)
        try:
            head_device = base_model.device
            if head_device != device:
                self.ea_layer.diff_device = True
                self.ea_layer.gen_head  = base_model.gen_head.to(device)
        except AttributeError:
            pass
        

    @classmethod
    def from_pretrained(
        cls,
        base_model_path:    str = "/work1/deming/shared/Janus-Pro-7B",
        drafter_model_path: str = "/work1/deming/shared/Eagle-Janus-Pro",
        top_k:        int   = 10,
        total_tokens: int   = 59,
        depth:        int   = 5,
        threshold:    float = 1.0,
        **kwargs,
    ) -> "Model":
        
        base_model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            base_model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, **kwargs
        ).cuda().eval()
        
        
        drafter_config = AutoConfig.from_pretrained(drafter_model_path)
        
        load_model_path = os.path.join(drafter_model_path, "model.safetensors")
        if not os.path.exists(load_model_path):
            load_model_path = hf_hub_download(drafter_model_path, "model.safetensors")
        ea_layer_state_dict = load_file(load_model_path)

        processor = VLChatProcessor.from_pretrained(base_model_path)
        model = cls(
            base_model,
            drafter_config,
            ea_layer_state_dict,
            processor,
            top_k, total_tokens, depth, threshold)

        return model

    def pad_path(self, path: List[int], length: int, pad_value: int = -2) -> List[int]:
        """
        Pad the given path list with a specific value up to a specified length.

        Parameters:
        - path (list): The original list that needs padding.
        - length (int): The desired length of the padded list.
        - pad_value (optional, default=-2): The value to use for padding.

        Returns:
        - list: A new list based on the original path but padded to the desired length.

        Example:
        >>> pad_path([1,2,3], 5)
        [1, 2, 3, -2, -2]

        Note:
        If the given path is already longer than the specified length,
        then no padding occurs, and the original path is returned.
        """

        # Calculate the number of padding values needed by subtracting the length
        # of the path from the desired length.
        # Append the padding values to the original path and return the new list.
        return path + [pad_value] * (length - len(path))
    
    # ------------------------------------------------------------------
    # Target forward — mirrors ea2_model_llamagen.py forward()
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids:       Optional[torch.Tensor] = None,
        inputs_embeds:   Optional[torch.Tensor] = None,
        attention_mask:  Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        output_orig:     bool = False,
        position_ids:    Optional[torch.Tensor] = None,
    ):
        """Target model forward.

        Accepts either input_ids (image tokens → prepare_gen_img_embeds)
        or inputs_embeds (prompt prefix, already embedded).
        """
        with torch.inference_mode():
            if inputs_embeds is None:
                # Image token phase — use prepare_gen_img_embeds
                # input_ids shape: (2*B, n_tokens) — cond + uncond interleaved
                inputs_embeds = self.base_model.prepare_gen_img_embeds(
                    input_ids.view(-1)
                ).view(input_ids.shape[0], input_ids.shape[1], -1)

            outputs = self.base_model.language_model.model(
                inputs_embeds   = inputs_embeds,
                attention_mask  = attention_mask,
                past_key_values = past_key_values,
                position_ids    = position_ids,
                use_cache       = True,
            )

            hidden_states = outputs.last_hidden_state

            if output_orig:
                # gen_head: (2*B, seq_len, V) → split cond/uncond → CFG
                orig = self.base_model.gen_head(hidden_states)
                return outputs, orig, hidden_states

        return outputs, hidden_states

    @torch.no_grad()
    def generate_tree_buffers(self, tree_choices, level_t, device="cuda"):
        sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
        if level_t is not None:
            sorted_tree_choices = [x for x in sorted_tree_choices if len(x) <= level_t]
        tree_len = len(sorted_tree_choices) + 1

        # Initialize depth_counts to keep track of how many choices have a particular depth
        depth_counts = []
        prev_depth = 0
        for path in sorted_tree_choices:
            depth = len(path)
            if depth != prev_depth:
                depth_counts.append(0)
            depth_counts[depth - 1] += 1
            prev_depth = depth

        tree_attn_mask = torch.eye(tree_len, tree_len)
        tree_attn_mask[:, 0] = 1
        start = 0
        for i in range(len(depth_counts)):
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                # retrieve ancestor position
                if len(cur_tree_choice) == 1:
                    continue
                ancestor_idx = []
                for c in range(len(cur_tree_choice) - 1):
                    ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
                tree_attn_mask[j + start + 1, ancestor_idx] = 1
            start += depth_counts[i]

        tree_indices = torch.zeros(tree_len, dtype=torch.long)
        p_indices = [0 for _ in range(tree_len - 1)]
        b_indices = [[] for _ in range(tree_len - 1)]
        tree_indices[0] = 0
        start = 0
        bias = 0
        for i in range(len(depth_counts)):
            inlayer_bias = 0
            b = []
            for j in range(depth_counts[i]):
                cur_tree_choice = sorted_tree_choices[start + j]
                cur_parent = cur_tree_choice[:-1]
                if j != 0:
                    if cur_parent != parent:
                        bias += 1
                        inlayer_bias += 1
                        parent = cur_parent
                        b = []
                else:
                    parent = cur_parent
                tree_indices[start + j + 1] = cur_tree_choice[-1] + TOPK * (i + bias) + 1
                p_indices[start + j] = inlayer_bias
                if len(b) > 0:
                    b_indices[start + j] = copy.deepcopy(b)
                else:
                    b_indices[start + j] = []
                b.append(cur_tree_choice[-1] + TOPK * (i + bias) + 1)
            start += depth_counts[i]

        p_indices = [-1] + p_indices
        tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
        start = 0
        for i in range(len(depth_counts)):
            tree_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
            start += depth_counts[i]

        retrieve_indices_nest = []
        retrieve_paths = []
        for i in range(len(sorted_tree_choices)):
            cur_tree_choice = sorted_tree_choices[-i - 1]
            retrieve_indice = []
            if cur_tree_choice in retrieve_paths:
                continue
            else:
                for c in range(len(cur_tree_choice)):
                    retrieve_indice.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]))
                    retrieve_paths.append(cur_tree_choice[:c + 1])
            retrieve_indices_nest.append(retrieve_indice)
        max_length = max([len(x) for x in retrieve_indices_nest])
        retrieve_indices = [self.pad_path(path, max_length) for path in retrieve_indices_nest]
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        retrieve_indices = retrieve_indices + 1
        retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
                                    dim=1)

        maxitem = retrieve_indices.max().item() + 5

        def custom_sort(lst):
            # sort_keys=[len(list)]
            sort_keys = []
            for i in range(len(lst)):
                sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
            return sort_keys

        retrieve_indices = retrieve_indices.tolist()
        retrieve_indices = sorted(retrieve_indices, key=custom_sort)
        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)

        p_indices = torch.tensor(p_indices)
        p_indices_new = p_indices[retrieve_indices]
        p_indices_new = p_indices_new.tolist()

        b_indices = [[]] + b_indices
        b_indices_new = []
        for ib in range(retrieve_indices.shape[0]):
            iblist = []
            for jb in range(retrieve_indices.shape[1]):
                index = retrieve_indices[ib, jb]
                if index == -1:
                    iblist.append([])
                else:
                    b = b_indices[index]
                    if len(b) > 0:
                        bt = []
                        for bi in b:
                            bt.append(torch.where(tree_indices == bi)[0].item())
                        iblist.append(torch.tensor(bt, device=device))
                    else:
                        iblist.append(b)
            b_indices_new.append(iblist)

        levels = torch.unique(tree_position_ids)
        per_level_node_counts = {int(level.item()): int((tree_position_ids == level).nonzero(as_tuple=True)[0][0]) for level in levels}

        # Aggregate the generated buffers into a dictionary
        tree_buffers = {
            "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),
            "tree_indices": tree_indices,
            "tree_position_ids": tree_position_ids,
            "retrieve_indices": retrieve_indices,
        }

        # Move the tensors in the dictionary to the specified device
        tree_buffers = {
            k: v.clone().to(device)
            if isinstance(v, torch.Tensor)
            else torch.tensor(v, device=device)
            for k, v in tree_buffers.items()
        }
        tree_buffers["p_indices"] = p_indices_new
        tree_buffers["b_indices"] = b_indices_new
        tree_buffers["per_level_node_counts"] = per_level_node_counts
        return tree_buffers, sorted_tree_choices
    
    def initialize_tree_v1(self, inputs_embeds, tokens, tree_choices, past_key_values, tree_attn_mask, temperature, top_k, top_p, cfg_scale=3.0):
        _, hidden_states = self(
            inputs_embeds   = inputs_embeds,
            past_key_values = past_key_values,
        )

        # First token
        logits_first  = self.base_model.gen_head(hidden_states[:, -1, :])   # (2*B, V)
            
        cfg_logits    = cfg_logit_process(logits_first, cfg_scale)           # (B, V)
        sample_token, _ = sample(cfg_logits, temperature, top_k, top_p)     # (B, 1)

        next_token = torch.cat([sample_token.unsqueeze(dim=1), sample_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = self.base_model.prepare_gen_img_embeds(next_token)
        inputs_embeds = torch.cat([inputs_embeds, img_embeds.unsqueeze(1)], dim=1)  # (2*B, prompt+1)
        
        # ------------------------------------------------------------------
        # 3. Init tree and first draft pass
        # ------------------------------------------------------------------
        self.ea_layer.init_tree_v1(tree_choices)
        self.base_model.tree_mask = tree_attn_mask
        
        tree_logits = self.ea_layer.topK_genrate(
            hidden_state     = hidden_states[:, -1:, :],
            image_embeds     = inputs_embeds[:, self.prompt_len:, :],
            head             = self.base_model.gen_head,
            cfg_scale        = cfg_scale,
            temperature      = temperature,
        )
        return tree_logits, cfg_logits, sample_token
       
    
    # ------------------------------------------------------------------
    # Tree decoding — single pass, mirrors tree_decoding() in LlamaGen
    # ------------------------------------------------------------------

    @torch.no_grad()
    def tree_decoding(
        self,
        tree_candidates:   torch.Tensor,   # (2*B, n_nodes) — cond+uncond interleaved
        past_key_values,
        tree_position_ids: torch.Tensor,
        input_ids:         torch.Tensor,
        retrieve_indices:  torch.Tensor,
        cfg_scale:         float,
        attention_mask:    Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, object]:
        """Verify all draft tokens in one target forward pass.

        Identical to LlamaGen tree_decoding but uses:
        - prepare_gen_img_embeds for image token embeddings
        - gen_head instead of lm_head
        - cfg_logit_process with even/odd split instead of batch-split
        """
        seq_len    = input_ids.shape[1] 
        position_ids = tree_position_ids + seq_len
        # position_ids = position_ids.view(-1, position_ids.shape[-1]).long()
        position_ids = position_ids.unsqueeze(0).expand(
            tree_candidates.shape[0], -1
        ).long()   # (2, n_nodes) — correct batch size

        if attention_mask is not None: # [2, 120]
            remaining_length = input_ids.shape[1] + tree_candidates.shape[1] - attention_mask.shape[1]
            one_padding = torch.ones((attention_mask.shape[0], remaining_length), dtype=torch.long, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, one_padding], dim=1)        
        
        # Embed tree candidates
        tree_embeds = self.base_model.prepare_gen_img_embeds(
            tree_candidates.view(-1)
        ).view(tree_candidates.shape[0], tree_candidates.shape[1], -1)
        
        outputs, tree_logits, hidden_state = self(
            inputs_embeds   = tree_embeds,
            output_orig     = True,
            past_key_values = past_key_values,
            position_ids    = position_ids,
            attention_mask  = attention_mask,
        )

        # Apply CFG (even=cond, odd=uncond)
        cfg_tree_logits = cfg_logit_process(tree_logits, cfg_scale)

        # Retrieve logits along candidate paths
        # cfg_tree_logits: (B, n_nodes, V) after CFG — use first batch element
        logits = cfg_tree_logits[0, retrieve_indices]   # (n_candidates, depth, V)

        return logits, hidden_state, outputs

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def generate_candidates(
        self,
        tree_logits,
        tree_indices,
        retrieve_indices,
        sample_token,
    ):
        """Identical to LlamaGen generate_candidates."""
        sample_token = sample_token.to(tree_indices.device)
        candidates   = torch.cat(
            [sample_token[0], tree_logits[0].view(-1)], dim=-1
        )
        tree_candidates = candidates[tree_indices]
        tree_candidates_ext = torch.cat(
            [tree_candidates,
             torch.zeros(1, dtype=torch.long, device=tree_candidates.device) - 1],
            dim=0
        )
        cart_candidates = tree_candidates_ext[retrieve_indices]

        candidates_prob = torch.cat(
            [torch.ones(1, device=tree_logits[1].device, dtype=torch.float32),
                tree_logits[1].view(-1)],
            dim=-1
        )
        tree_prob_ext = torch.cat(
            [candidates_prob[tree_indices],
                torch.ones(1, dtype=torch.float32, device=candidates_prob.device)],
            dim=0
        )
        cart_candidates_prob = tree_prob_ext[retrieve_indices]
        
         # Unsqueeze the tree candidates for dimension consistency.
        tree_candidates = tree_candidates.unsqueeze(0)
        return cart_candidates, cart_candidates_prob, tree_candidates

    # ------------------------------------------------------------------
    # Posterior evaluation
    # ------------------------------------------------------------------

    def evaluate_posterior(
        self,
        # bias_list,
        l_node_counts,
        logits,
        tree_logits,
        temperature,
        retrieve_indices,
        candidates,
        cart_candidates_prob,
        original_prob,
        p_indices,
        tree_candidates,
        b_indices,
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """Accept/reject draft tokens via speculative sampling.

        Mirrors evaluate_posterior_v1 from ea2_model_llamagen.py.
        Janus-Pro specific: hard-reject tokens outside image token range.
        """
        accept_length  = 1
        accept_cand    = candidates[0][:1]
        best_candidate = 0

        cart_candidates_prob = cart_candidates_prob.to(logits.device)

        for i in range(1, candidates.shape[1]):
            if i != accept_length:
                break

            adjustflag = False
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi    = torch.nonzero(is_eq, as_tuple=True)[0][0]

            gt_logits = logits[fi, i - 1]
            gtp = torch.softmax(gt_logits / temperature, dim=-1)

            m_bias_list = None
            past_nodes  = l_node_counts[i]
            # if i < len(bias_list) and len(bias_list[i]):
            #     m_bias_list = copy.deepcopy(bias_list[i])
            #     m_bias_list = [
            #         (a + past_nodes, b + past_nodes, dp, cs)
            #         for a, b, dp, cs in m_bias_list
            #     ]

            candidates_set = []

            for j in range(candidates.shape[0]):
                if not is_eq[j]:
                    continue

                x  = candidates[j, i]
                xi = x.item()

                if xi in candidates_set or xi == -1:
                    continue
                candidates_set.append(xi)

                r  = random.random()
                px = gtp[xi]

                qx = cart_candidates_prob[j, i]
                if qx <= 0:
                    continue

                # Neighbor bias boost from bias_list
                if px < qx and m_bias_list is not None:
                    curr_node = retrieve_indices[j, i]
                    for a, b, draft_prob, cosine_sim in m_bias_list:
                        if a == curr_node:
                            similar_xi = tree_candidates[0][b]
                            px = min(qx, px + r * gtp[similar_xi])
                            break

                acp = px / qx
                if r <= acp:
                    accept_cand    = torch.cat((accept_cand, x[None]), dim=0)
                    accept_length += 1
                    best_candidate = j
                    break
                else:
                    # Standard EAGLE residual adjustment
                    q = original_prob[i - 1][p_indices[j][i]].clone()
                    b = b_indices[j][i]
                    if len(b) > 0:
                        mask   = tree_candidates[0][b]
                        q[mask]= 0
                        q      = q / q.sum()

                    gtp = gtp - q
                    gtp[gtp < 0] = 0
                    if gtp.sum() == 0:
                        gtp = torch.ones_like(gtp)
                    gtp    = gtp / gtp.sum()
                    adjustflag = True

        if adjustflag and accept_length != candidates.shape[1]:
            sample_p = gtp 
        else:
            gt_logits = logits[best_candidate, accept_length - 1]
            sample_p  = torch.softmax(gt_logits / temperature, dim=-1)

        return torch.tensor(best_candidate), accept_length - 1, sample_p

    # ------------------------------------------------------------------
    # Update inference inputs — mirrors update_inference_inputs() LlamaGen
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_inference_inputs(
        self,
        idx,
        input_ids:         torch.Tensor,
        inputs_embeds:     torch.Tensor,
        candidates:        torch.Tensor,
        best_candidate:    torch.Tensor,
        accept_length:     int,
        retrieve_indices:  torch.Tensor,
        new_token:         int,
        past_key_values_data_list,
        current_length_data,
        hidden_state_new:  torch.Tensor,
        sample_p:          torch.Tensor,
        cfg_scale:         float,
        static_tree:       bool = True,
        tree_buffers:      dict = None,
        tree_choices:      list = None,
    ):
        prev_input_len = input_ids.shape[1]

        select_indices = (
            retrieve_indices[best_candidate, :accept_length + 1] + prev_input_len
        )

        # Append accepted tokens — input_ids is (2, T) for cond+uncond
        accepted_toks = candidates[None, best_candidate, :accept_length + 1]
        accepted_toks = accepted_toks.repeat(2, 1)
        accepted_toks_embed = self.base_model.prepare_gen_img_embeds(accepted_toks)
        input_ids     = torch.cat([input_ids, accepted_toks], dim=-1)
        inputs_embeds = torch.cat([inputs_embeds, accepted_toks_embed], dim = 1) # sequence dimension

        # Update KV cache in-place
        for past_kv_data in past_key_values_data_list:
            tgt = past_kv_data[..., select_indices.to(past_kv_data.device), :]
            dst = past_kv_data[..., prev_input_len:prev_input_len + tgt.shape[-2], :]
            dst.copy_(tgt, non_blocking=True)

        current_length_data.fill_(prev_input_len + tgt.shape[-2])

        # Extract accepted hidden state
        retrieve_hs  = hidden_state_new[:, retrieve_indices]
        accept_hs    = retrieve_hs[:, best_candidate, :accept_length + 1]

        # Sample bonus token
        token = torch.multinomial(sample_p, 1)[None]   # (1, 1)
        token = token.repeat(2, 1)
        new_token_embed = self.base_model.prepare_gen_img_embeds(token)
        ea_inputs_embeds = torch.concat([inputs_embeds, new_token_embed], dim = 1) # seq dimension
        
        tree_logits = self.ea_layer.topK_genrate(
            hidden_state     = accept_hs[:, -1:, :],
            image_embeds     = ea_inputs_embeds[:, self.prompt_len:, :],
            head             = self.base_model.gen_head,
            cfg_scale        = cfg_scale,
        )
        new_token += accept_length + 1
        return input_ids, tree_logits, new_token, token, inputs_embeds

    # ------------------------------------------------------------------
    # Main generate loop — mirrors generate() from ea2_model_llamagen.py
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt:       str,
        max_length:   int   = 576,
        temperature:  float = 1.0,
        top_k:        int   = 2000,
        top_p:        float = 1.0,
        cfg_scale:    float = 5.0,
        parallel_size:int   = 1,
        img_size:     int   = 384,
        patch_size:   int   = 16,
        tree_choices: list  = naive_extend_57,
    ) -> Tuple[torch.Tensor, float, List[int]]:
        """Generate image tokens with EAGLE speculative decoding.

        Returns:
            generated_tokens: (parallel_size, max_length) int64
            latency:          float seconds
            accept_list:      list[int] accepted tokens per step
        """

        device = "cuda"

        # Logits processor: top-k/top-p filter over image token range
        # def logits_processor(_, logits):
        #     if logits is None:
        #         return logits
        #     out = mask_non_image_logits(logits.clone())
        #     return top_k_top_p_filtering(out, top_k=top_k, top_p=top_p)

        # ------------------------------------------------------------------
        # 1. Build prompt input — matches official Janus-Pro inference exactly
        # ------------------------------------------------------------------

        conversation = [
            {"role": "<|User|>",      "content": prompt[0]},
            {"role": "<|Assistant|>", "content": ""},
        ]
        sft_format = self.processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.processor.sft_format,
            system_prompt="",
        )
        prompt_str = sft_format + self.processor.image_start_tag
        input_ids_raw = self.processor.tokenizer.encode(prompt_str)
        input_ids_raw = torch.LongTensor(input_ids_raw)

        # (2*parallel_size, prompt_len) — even=cond, odd=uncond
        B2 = parallel_size * 2
        tokens = torch.zeros((B2, len(input_ids_raw)), dtype=torch.int, device=device)
        for i in range(B2):
            tokens[i, :] = input_ids_raw
            if i % 2 != 0:
                # Unconditional: mask prompt tokens
                tokens[i, 1:-1] = self.processor.pad_id

        # Embed prompt
        inputs_embeds = self.base_model.language_model.get_input_embeddings()(tokens)  
        input_ids = tokens     
        prompt_len =  inputs_embeds.shape[1]
        self.prompt_len = prompt_len
            
        # ------------------------------------------------------------------
        # 2. Run prefix forward pass to fill KV cache in initialize_tree_v1
        # ------------------------------------------------------------------
        past_key_values, past_key_values_data, current_length_data = januspro_initialize_past_key_values(self.base_model)
        self.base_model.past_key_values = past_key_values
        self.base_model.past_key_values_data = past_key_values_data
        self.base_model.current_length_data = current_length_data
            
        tree_buffers, _ = self.generate_tree_buffers(
            tree_choices, None, device="cuda"
        )
        tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(self.base_model.device)
        self.tree_choices = tree_choices
        self.tree_buffers = tree_buffers
        
        tree_logits, logits, sample_token = self.initialize_tree_v1(
            inputs_embeds, tokens, tree_choices, past_key_values, tree_buffers['tree_attn_mask'], temperature, top_k, top_p, cfg_scale
        )
         # generated_tokens accumulates accepted image tokens
        generated_tokens = torch.zeros(
            (1, max_length), dtype=torch.int, device="cuda"
        )
        # generated_tokens[:, 0] = sample_token.squeeze(-1)
            
        # Scale KV cache batch dim to parallel_size * 2
        if parallel_size > 1:
            past_key_values_data = [
                t.repeat(parallel_size, 1, 1, 1) for t in past_key_values_data
            ]

        new_token  = 0
        accept_list = []

        # ------------------------------------------------------------------
        # 4. Speculative decoding loop
        # ------------------------------------------------------------------
        st = time.time()
        for idx in range(1, max_length):

            candidates, cart_candidates_prob, tree_candidates = self.generate_candidates(
                tree_logits,
                tree_buffers["tree_indices"],
                tree_buffers["retrieve_indices"],
                sample_token,
            )

            # Duplicate candidates for cond + uncond
            tree_candidates_cfg = tree_candidates.repeat(2, 1)
            

            logits, hidden_state_new, _ = self.tree_decoding(
                tree_candidates_cfg,
                past_key_values,
                tree_buffers["tree_position_ids"],
                input_ids,
                tree_buffers["retrieve_indices_head"],
                cfg_scale,
            )

            best_candidate, accept_length, sample_p = self.evaluate_posterior(
                # bias_list,
                tree_buffers["per_level_node_counts"],
                logits,
                tree_logits,
                temperature,
                tree_buffers["retrieve_indices"],
                candidates,
                cart_candidates_prob,
                tree_logits[2],
                tree_buffers["p_indices"],
                tree_candidates,
                tree_buffers["b_indices"],
            )
            # accept_length = 0
            # best_candidate = 0
            # sample_p = torch.softmax(logits[0, 0] / temperature, dim=-1)
            
            accept_list.append(accept_length)
            # Store accepted tokens
            accepted_toks = candidates[None, best_candidate, :accept_length + 1] 
            end_pos = min(new_token + accept_length + 1, max_length)
            fill_len = end_pos - new_token
            generated_tokens[:, new_token:end_pos] = accepted_toks[:, :fill_len]
            input_ids, tree_logits, new_token, sample_token, inputs_embeds = \
                self.update_inference_inputs(
                    idx,
                    input_ids,
                    inputs_embeds,
                    candidates,
                    best_candidate,
                    accept_length,
                    tree_buffers["retrieve_indices"],
                    new_token,
                    past_key_values_data,
                    current_length_data,
                    hidden_state_new,
                    sample_p,
                    cfg_scale,
                    static_tree=True,
                    tree_buffers=tree_buffers,
                    tree_choices=tree_choices,
                )

            if new_token >= max_length:
                break

        return generated_tokens, time.time() - st, accept_list