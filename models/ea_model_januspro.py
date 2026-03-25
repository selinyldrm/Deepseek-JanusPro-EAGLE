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
    - Image tokens at vocab indices [IMAGE_TOKEN_START, IMAGE_TOKEN_END)
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
from janus.models import MultiModalityCausalLM, VLChatProcessor
from models.drafters.choices import naive_extend_57

import copy
import json
import time
from typing import List, Optional

from torchvision.utils import save_image
import math
import matplotlib.pyplot as plt

from PIL import Image

from .drafters.utils import *
from .drafters.kv_cache import initialize_past_key_values

from .drafters.cnets_januspro import Model, cfg_logit_process
from .configs.configs import EConfig

from .drafters.choices import *

import torch.nn.functional as F
import copy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_TOKEN_START = 100015
IMAGE_TOKEN_END   = 116399   # exclusive
TOPK = 10                    # default draft top-k, overridden by EaModel.top_k


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def mask_non_image_logits(logits: torch.Tensor) -> torch.Tensor:
    """Zero out all vocab positions outside image token range."""
    logits[..., :IMAGE_TOKEN_START] = -float("Inf")
    logits[..., IMAGE_TOKEN_END:]   = -float("Inf")
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
# KV cache (mirrors kv_cache.py from LlamaGen)
# ---------------------------------------------------------------------------

def initialize_past_key_values(model: MultiModalityCausalLM):
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

    past_key_values_data = []
    past_key_values      = []
    current_length_data  = torch.zeros(1, dtype=torch.long, device=device)

    for _ in range(num_layers):
        k = torch.zeros(batch_size, num_kv_heads, max_length, head_dim,
                        device=device, dtype=dtype)
        v = torch.zeros(batch_size, num_kv_heads, max_length, head_dim,
                        device=device, dtype=dtype)
        past_key_values_data.extend([k, v])
        past_key_values.append([k, v])

    return past_key_values, past_key_values_data, current_length_data


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
        drafter_model: MultiModalityCausalLM,
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
            model         = drafter_model.eval(),
            top_k         = top_k,
            depth         = depth,
            total_tokens  = total_tokens,
            threshold     = threshold,
        )

        device = "cuda"
        # Check if gen_head is on a different device (model parallelism)
        try:
            head_device = base_model.gen_head.weight.device
            if head_device != device:
                self.ea_layer.diff_device = True
                self.ea_layer.headweight  = base_model.gen_head.weight.clone().to(device)
        except AttributeError:
            pass

        self.ea_layer.to(base_model.language_model.dtype).to(device)
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        base_model_path:    str = "/work1/deming/shared/Janus-Pro-7B",
        drafter_model_path: str = "/work1/deming/shared/Janus-Pro-1B",
        top_k:        int   = 10,
        total_tokens: int   = 59,
        depth:        int   = 5,
        threshold:    float = 1.0,
        **kwargs,
    ) -> "Model":
        print(f"Loading target:  {base_model_path}")
        
        base_model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            base_model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, **kwargs
        ).cuda().eval()

        print(f"Loading drafter: {drafter_model_path}")
        
        drafter: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            drafter_model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, **kwargs
        ).cuda().eval()

        processor = VLChatProcessor.from_pretrained(base_model_path)

        return cls(base_model, drafter, processor,
                   top_k, total_tokens, depth, threshold)

    # ------------------------------------------------------------------
    # Target forward — mirrors ea2_model_llamagen.py forward()
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids:       Optional[torch.Tensor] = None,
        inputs_embeds:   Optional[torch.Tensor] = None,
        attention_mask:  Optional[torch.Tensor] = None,
        past_key_values = None,
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

        if attention_mask is not None:
            n_extra = tree_candidates.shape[1]
            extra   = torch.ones(
                (attention_mask.shape[0], n_extra),
                dtype=torch.long, device=attention_mask.device
            )
            attention_mask = torch.cat([attention_mask, extra], dim=1)

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
        logits_processor,
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

        if logits_processor is not None:
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
        else:
            cart_candidates_prob = None

        return cart_candidates, cart_candidates_prob, tree_candidates.unsqueeze(0)

    # ------------------------------------------------------------------
    # Posterior evaluation
    # ------------------------------------------------------------------

    def evaluate_posterior(
        self,
        bias_list,
        l_node_counts,
        logits,
        tree_logits,
        retrieve_indices,
        candidates,
        cart_candidates_prob,
        original_prob,
        p_indices,
        tree_candidates,
        b_indices,
        logits_processor=None,
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

            gt_logits = logits[fi, i - 1][None]
            if logits_processor is not None:
                gt_logits = logits_processor(None, gt_logits)
            gt_logits = mask_non_image_logits(gt_logits[0].clone())
            gtp = torch.softmax(gt_logits, dim=0)

            m_bias_list = None
            past_nodes  = l_node_counts[i]
            if i < len(bias_list) and len(bias_list[i]):
                m_bias_list = copy.deepcopy(bias_list[i])
                m_bias_list = [
                    (a + past_nodes, b + past_nodes, dp, cs)
                    for a, b, dp, cs in m_bias_list
                ]

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

                # Hard-reject non-image tokens
                if not (IMAGE_TOKEN_START <= xi < IMAGE_TOKEN_END):
                    px = torch.tensor(0.0, device=logits.device)

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
            gt_logits = logits[best_candidate, accept_length - 1][None]
            if logits_processor is not None:
                gt_logits = logits_processor(None, gt_logits)
            gt_logits = mask_non_image_logits(gt_logits[0].clone())
            sample_p  = torch.softmax(gt_logits, dim=0)

        return torch.tensor(best_candidate), accept_length - 1, sample_p

    # ------------------------------------------------------------------
    # Update inference inputs — mirrors update_inference_inputs() LlamaGen
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_inference_inputs(
        self,
        idx,
        input_ids:         torch.Tensor,
        candidates:        torch.Tensor,
        best_candidate:    torch.Tensor,
        accept_length:     int,
        retrieve_indices:  torch.Tensor,
        logits_processor,
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
        input_ids     = torch.cat([input_ids, accepted_toks.repeat(2, 1)], dim=-1)

        # Update KV cache in-place
        for past_kv_data in past_key_values_data_list:
            tgt = past_kv_data[..., select_indices.to(past_kv_data.device), :]
            dst = past_kv_data[..., prev_input_len:prev_input_len + tgt.shape[-2], :]
            dst.copy_(tgt, non_blocking=True)

        current_length_data.fill_(prev_input_len + accept_length + 1)

        # Extract accepted hidden state
        retrieve_hs  = hidden_state_new[:, retrieve_indices]
        accept_hs    = retrieve_hs[:, best_candidate, :accept_length + 1]

        # Sample bonus token
        token = torch.multinomial(sample_p, 1)[None]   # (1, 1)

        # Build drafter input — (2, T+1) cond+uncond with bonus token
        ea_input_ids = torch.cat(
            (input_ids, token.repeat(2, 1).to(input_ids.device)), dim=1
        )

        if static_tree:
            tree_logits = self.ea_layer.topK_genrate(
                hidden_state     = accept_hs,
                input_ids        = ea_input_ids,
                head             = self.base_model.gen_head,
                logits_processor = logits_processor,
                cfg_scale        = cfg_scale,
            )
            new_token += accept_length + 1
            return input_ids, tree_logits, new_token, None, token

        # Dynamic tree not implemented here — extend if needed
        raise NotImplementedError("Dynamic tree not yet implemented for Janus-Pro")

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
        print("input tokens.shape: ", tokens.shape, flush=True)
        for i in range(B2):
            tokens[i, :] = input_ids_raw
            if i % 2 != 0:
                # Unconditional: mask prompt tokens
                tokens[i, 1:-1] = self.processor.pad_id

        # Embed prompt
        inputs_embeds = self.base_model.language_model.get_input_embeddings()(tokens)
        print("inputs_embeds.shape: ", inputs_embeds.shape, flush=True)
        # ------------------------------------------------------------------
        # 2. Run prefix forward pass to fill KV cache
        # ------------------------------------------------------------------
        past_key_values, past_key_values_data, current_length_data = \
            initialize_past_key_values(self.base_model)

       
        # Scale KV cache batch dim to parallel_size * 2
        if parallel_size > 1:
            past_key_values_data = [
                t.repeat(parallel_size, 1, 1, 1) for t in past_key_values_data
            ]

        prefix_out = self.base_model.language_model.model(
            inputs_embeds   = inputs_embeds,
            past_key_values = past_key_values,
            use_cache       = True,
        )

        hidden_states = prefix_out.last_hidden_state   # (2*B, prompt_len, D)
        print("hidden_states.shape: ", hidden_states.shape, flush=True)
        past_key_values = prefix_out.past_key_values

        # First token
        logits_first  = self.base_model.gen_head(hidden_states[:, -1, :])   # (2*B, V)
        print("logits_first.shape: ", logits_first.shape, flush=True)
        cfg_logits    = cfg_logit_process(logits_first, cfg_scale)           # (B, V)
        sample_token, _ = sample(cfg_logits, temperature, top_k, top_p)     # (B, 1)

        # generated_tokens accumulates accepted image tokens
        generated_tokens = torch.zeros(
            (parallel_size, max_length), dtype=torch.int, device=device
        )
        generated_tokens[:, 0] = sample_token.squeeze(-1)

        # Build input_ids for the generation loop — (2, prompt_len + 1)
        # Append first token in cond+uncond pattern
        first_tok_cfg = sample_token.repeat_interleave(2, dim=0)   # (2*B, 1)
        input_ids_gen = torch.cat([tokens, first_tok_cfg], dim=1)  # (2*B, prompt+1)

        # ------------------------------------------------------------------
        # 3. Init tree and first draft
        # ------------------------------------------------------------------
        tree_buffers = self.ea_layer._build_tree_buffers(tree_choices, device)
        self.ea_layer.init_tree(tree_choices, device)

        tree_logits = self.ea_layer.topK_genrate(
            hidden_state     = hidden_states,
            input_ids        = input_ids_gen,
            head             = self.base_model.gen_head,
            cfg_scale        = cfg_scale,
        )
        print("tree_logits.shape: ", tree_logits.shape, flush=True)

        new_token  = 1
        accept_list = []

        # ------------------------------------------------------------------
        # 4. Speculative decoding loop
        # ------------------------------------------------------------------
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
                input_ids_gen,
                tree_buffers["retrieve_indices_head"],
                cfg_scale,
            )

            best_candidate, accept_length, sample_p = self.evaluate_posterior(
                bias_list,
                tree_buffers["per_level_node_counts"],
                logits,
                tree_logits,
                tree_buffers["retrieve_indices"],
                candidates,
                cart_candidates_prob,
                tree_logits[2],
                tree_buffers["p_indices"],
                tree_candidates,
                tree_buffers["b_indices"],
            )

            accept_list.append(accept_length)

            # Store accepted tokens
            accepted_toks = candidates[best_candidate, :accept_length + 1]
            end_pos = min(new_token + accept_length + 1, max_length)
            fill_len = end_pos - new_token
            generated_tokens[:, new_token:end_pos] = accepted_toks[:fill_len].unsqueeze(0)

            input_ids_gen, tree_logits, new_token, _, sample_token, bias_list, _ = \
                self.update_inference_inputs(
                    idx,
                    input_ids_gen,
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

    # ------------------------------------------------------------------
    # Decode image tokens to pixel image — mirrors decode_ids() in LlamaGen
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_ids(
        self,
        generated_tokens: torch.Tensor,   # (B, max_length)
        img_size:   int = 384,
        patch_size: int = 16,
    ) -> np.ndarray:
        """Decode VQ token ids to RGB image array.

        Uses gen_vision_model.decode_code — same as official inference.
        """
        B     = generated_tokens.shape[0]
        codes = generated_tokens.to(dtype=torch.int)

        dec = self.base_model.gen_vision_model.decode_code(
            codes,
            shape=[B, 8, img_size // patch_size, img_size // patch_size],
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
        return dec