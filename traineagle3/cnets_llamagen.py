import copy
import os
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from models.configs.configs import EConfig    
from models.drafters.choices import *
from torch.utils.checkpoint import checkpoint

from traineagle3.modeling_llamagen_kv import LlamaForCausalLM as KVLlamaForCausalLM



TOPK=10

def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    """
    Pad the given path list with a specific value up to a specified length.
    """
    return path + [pad_value] * (length - len(path))

class node:
    def __init__(self,parent=None,value=None,dict_key=None):
        self.parent=parent
        self.value=value
        if parent:
            self.depth=parent.depth+1
            parent.children.append(self)
        else:
            self.depth=0
        self.children=[]
        self.dict_key=dict_key
    def is_leaf(self):
        return len(self.children)==0

    def all_index(self):
        if not self.parent.parent:
            return [self.index]
        else:
            return self.parent.all_index()+[self.index]

class Tree:
    def __init__(self,tree_list):
        sorted_tree_list = sorted(tree_list, key=lambda x: (len(x), x))
        self.root=node()
        self.node_dic={}
        for tree_node in sorted_tree_list:
            cur_value=tree_node[-1]
            if len(tree_node)==1:
                cur_node=node(parent=self.root,value=cur_value,dict_key=tuple(tree_node))
            else:
                cur_parent=self.node_dic[tuple(tree_node[:-1])]
                cur_node = node(parent=cur_parent, value=cur_value,dict_key=tuple(tree_node))
            self.node_dic[tuple(tree_node)] = cur_node
        self.indexnode()

    def max_depth(self):
        return max([item.depth for item in self.node_dic.values()])

    def num_node_wchild(self):
        num_c=0
        for item in self.node_dic.values():
            if not item.is_leaf():
                num_c+=1
        return num_c

    def get_node_wchild(self):
        ns=[]
        for item in self.node_dic.values():
            if not item.is_leaf():
                ns.append(item)
        return ns

    def indexnode(self):
        cur_index=0
        for key in self.node_dic:
            cur_node=self.node_dic[key]
            if not cur_node.is_leaf():
                cur_node.index=cur_index
                cur_index+=1

def generate_tree_buffers(tree_choices, device="cuda"):
    tree=Tree(tree_choices)
    sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
    tree_len = tree.num_node_wchild()

    max_depth=tree.max_depth()
    nodes_wc=tree.get_node_wchild()

    depth_counts=[0 for _ in range(max_depth-1)]
    for x in nodes_wc:
        depth_counts[x.depth-1]+=1
    depth_counts_sum = [sum(depth_counts[:i + 1]) for i in range(len(depth_counts))]

    tree_attn_mask = torch.eye(tree_len, tree_len)

    for id,x in enumerate(nodes_wc):
        tree_attn_mask[id,x.all_index()]=1

    tree_attn_mask_list0=[tree_attn_mask[:ml,:ml] for ml in depth_counts_sum]
    tree_attn_mask_list=[]
    for id,x in enumerate(tree_attn_mask_list0):
        x=x[-depth_counts[id]:]
        tree_attn_mask_list.append(x)

    tree_indices_list = [torch.zeros(ml, dtype=torch.long) for ml in depth_counts]
    repeat_nums=[[] for _ in depth_counts]
    start = 0
    bias = 0
    for i in range(len(depth_counts)):
        bias = 0
        repeat_j=0
        for j in range(depth_counts[i]):
            cur_node = nodes_wc[start + j]
            cur_parent = cur_node.parent

            if j != 0:
                if cur_parent != parent:
                    bias += 1
                    parent = cur_parent
                    repeat_nums[i].append(j-repeat_j)
                    repeat_j=j
            else:
                parent = cur_parent
            tree_indices_list[i][j] = cur_node.value + TOPK * (bias)
        repeat_nums[i].append(j - repeat_j+1)
        start += depth_counts[i]

    position_ids = [torch.zeros(ml, dtype=torch.long) for ml in depth_counts]

    tree_buffers = {
        "attn_mask": [i.unsqueeze(0).unsqueeze(0) for i in tree_attn_mask_list],
        "tree_indices": tree_indices_list,
        "position_ids":position_ids,
        "repeat_nums":repeat_nums
    }

    # Move the tensors in the dictionary to the specified device
    tree_buffers = {
        k: [i.clone().to(device) for i in v]
        if isinstance(v[0], torch.Tensor)
        else (
            torch.tensor(v, device=device)
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in tree_buffers.items()
    }
    return tree_buffers

class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)



# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

# LlamaGen specific functions and classes
def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs) # (grid_size, head_dim // 2)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    if cls_token_num > 0:
        cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
        cond_cache = torch.cat([cond_cache, torch.zeros(10, n_elem // 2, 2)])
    else:
        cond_cache = torch.cat([cache, torch.zeros(10, n_elem // 2, 2)])
    return cond_cache 


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.pretraining_tp = config.pretraining_tp
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        # Eagle 3: Modified to accept concatenated input features
        self.q_proj = nn.Linear(self.hidden_size * 2, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            if hasattr(self.config, "rope_theta"):
                self.rotary_emb = LlamaRotaryEmbedding(self.head_dim,
                                                       max_position_embeddings=self.max_position_embeddings,
                                                       base=self.config.rope_theta)
            else:
                self.rotary_emb = LlamaRotaryEmbedding(self.head_dim,
                                                       max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            cache_hidden: Optional[List[torch.Tensor]] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            freqs_cis: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        lck = len(cache_hidden[0])

        # cache_k = [self.k_proj(hidden) for hidden in cache_hidden]
        # cache_v = [self.v_proj(hidden) for hidden in cache_hidden]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)


        query_states = apply_rotary_emb(query_states.transpose(1, 2), freqs_cis.to(query_states.device)).transpose(1, 2)
        key_states = apply_rotary_emb(key_states.transpose(1, 2), freqs_cis.to(key_states.device)).transpose(1, 2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)


        # Avoid modify hidden cache inplace which will cause in-place modification error when enable gradient checkpoint. 
        # Return the updated hidden cache instead.
        if cache_hidden is None:
            local_cache_k = []
            local_cache_v = []
        else:
            local_cache_k = list(cache_hidden[0])
            local_cache_v = list(cache_hidden[1])

        local_cache_k.append(key_states)
        local_cache_v.append(value_states)
            
        cache_k = local_cache_k
        cache_v = local_cache_v

        k0 = cache_k[0]
        v0 = cache_v[0]

        attn_weights = torch.matmul(query_states, k0.transpose(2, 3)) / math.sqrt(self.head_dim)
        lck = len(cache_k)


        attn_weights = attn_weights + attention_mask
        

        for i in range(1, lck):
            ki = cache_k[i]

            qi = query_states
            kiq = ki

            attn_weightsi = (qi * kiq).sum(-1) / math.sqrt(self.head_dim)
            attn_weights = torch.cat((attn_weights, attn_weightsi[..., None]), dim=-1)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights0 = attn_weights[..., :q_len]
        
        attn_output = torch.matmul(attn_weights0, v0)

        for i in range(1, lck):
            vi = cache_v[i]
            attn_weightsi = attn_weights[..., q_len + i - 1]
            attn_outputi = attn_weightsi[..., None] * vi
            attn_output = attn_output + attn_outputi

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)


        attn_output = self.o_proj(attn_output)

        # Return the updated hidden cache.
        new_past_key_value = [local_cache_k,local_cache_v]
        return attn_output, new_past_key_value

class LlamaMLP(nn.Module):
    def __init__(self, config, last=True):
        super().__init__()
        self.last = last
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # if last:
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # else:
        #     self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size * 2, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaDecoderLayeremb(nn.Module):
    def __init__(self, config, last=True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config, last=last)
        self.last = last
        # self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: Optional[List[torch.Tensor]] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Eagle 3 Style Decoder Layer with input embedding fusion for LlamaGen
        """
        residual = hidden_states

        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        
        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)

        return_hidden = hidden_states

        # cache_hidden.append(hidden_states)


        # Self Attention
        hidden_states, latest_hidden_cache = self.self_attn(
            cache_hidden=cache_hidden,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            freqs_cis=freqs_cis,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states


        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, return_hidden)


        return outputs, latest_hidden_cache
    
@torch.no_grad()
def padding(tensor, left=True):
    """Helper function for padding tensors"""
    zeropadding = torch.zeros_like(tensor[:, -1:]) # (batch_size, 1)
    if left:
        tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    else:
        tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    return tensor

def len_list(x, n):
    return [i for i in x if len(i) <= n]

def repeat_hidden(hidden_states, num_repeat):
    new_hidden = []
    for id, i in enumerate(num_repeat):
        new_hidden.append(hidden_states[:, id:id+1].repeat(1, i, 1))
    return torch.cat(new_hidden, dim=1)

def sample(logits, k=1):
    # logits : logits after logit processors
    # k : number of samples to be sampled

    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    sampled_indices = torch.multinomial(probabilities, k, replacement=False)
    sampled_probs = torch.gather(probabilities, -1, sampled_indices)

    cumulative_sum = torch.cumsum(sampled_probs, dim=-1)
    cumulative_sum = torch.cat(
        (torch.zeros(cumulative_sum.shape[0], 1, device=cumulative_sum.device), cumulative_sum[:, :-1]), dim=-1
    )

    sampled_probs = sampled_probs / (1 - cumulative_sum) # probability normalization?
    sampled_probs[torch.isinf(sampled_probs)] = -1
    sampled_probs[torch.isnan(sampled_probs)] = -1

    sampled_probs = torch.clamp(sampled_probs, min=0.0, max=1.0)

    return sampled_indices, sampled_probs, probabilities

class Model(nn.Module):
    """
    Eagle 3 Model for LlamaGen with Multi-Layer Feature Extraction
    """
    def __init__(self, config, training_config, load_emb=False, path=None, bias=True, total_tokens=63, depth=5, top_k=8, threshold=1.0, embed_upscale=1.0):
        super().__init__()

        self.config = config
        self.training_config = training_config
        self.gradient_checkpointing = False
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.length = training_config.length  # Eagle 3 multi-step training length = 7

        self.lm_head=nn.Linear(config.hidden_size,config.vocab_size,bias=False)

        # Load target model for multi-layer feature extraction
        if path is not None:
            # [SY]: fix base model import 
            # self.target_model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.float16, output_hidden_states=True)
            self.target_model = KVLlamaForCausalLM.from_pretrained(
                path
            )
            self.target_model.eval()
            for param in self.target_model.parameters():
                param.requires_grad = False
            # Use target model's hidden size and vocab size for feature dimensions
            self.target_hidden_size = self.target_model.config.hidden_size
            self.target_vocab_size = self.target_model.config.vocab_size

            
            state_dict = torch.load(path + "/pytorch_model.bin", map_location="cpu")

            # If it's saved under "state_dict", unwrap it
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            tensor = state_dict["model.embed_tokens.weight"]
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx, _weight=tensor.float())
            for param in self.embed_tokens.parameters():
                param.requires_grad = False

            self.lm_head.weight = self.target_model.lm_head.weight 
            for param in self.lm_head.parameters():
                param.requires_grad = False

        else:
            self.target_model = None
            self.target_hidden_size = config.hidden_size
            self.target_vocab_size = config.vocab_size

        self.top_k = top_k
        self.total_tokens = total_tokens - 1
        self.depth = depth
        self.threshold = math.log(threshold)
        self.embed_upscale = embed_upscale

        # Eagle 3: Single decoder layer for multi-step processing
        self.midlayer = LlamaDecoderLayeremb(config)
        
        # Eagle 3: 3-feature input fusion (hidden_states from 3 layers)
        # Use target model's hidden size instead of config hidden size
        self.fc = nn.Linear(3 * self.target_hidden_size, config.hidden_size, bias=False)
        
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        
        # LlamaGen specific configurations
        config.block_size = 1024
        config.rope_base = 10000
        config.cls_token_num = 119
    
        # if hasattr(config, 'input_type'):
        #     if config.input_type == "c2i":
        #         config.block_size = 576
        #         config.rope_base = 10000
        #         config.cls_token_num = 0
        #     elif config.input_type == "t2i":
        #         config.block_size = 256
        #         config.rope_base = 10000
        #         config.cls_token_num = 119
        #     elif config.input_type == "t2i2":
        #         config.block_size = 1024
        #         config.rope_base = 10000
        #         config.cls_token_num = 119
            
        grid_size = int(config.block_size ** 0.5)
        assert grid_size ** 2 == config.block_size, "block_size must be a perfect square"
        # Use target model config for correct head dimensions
        # For Eagle3, the head_dim should match the actual attention head dimension
        head_dim = config.hidden_size // config.num_attention_heads
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, head_dim, config.rope_base, config.cls_token_num)
        # expand the freqs_cis for few steps to avoid out of bound error
        padding_freqs = torch.zeros_like(self.freqs_cis[:10])
        self.freqs_cis = torch.cat([self.freqs_cis, padding_freqs], dim=0)
        
        

    @torch.no_grad()
    def dataprepare(self, cond_idx, input_ids, attention_mask, loss_mask):
        """
        Eagle 3: Extract features from 3 different layers of the target model (LlamaGen)
        """
        device = input_ids.device
        
        # Get outputs with hidden states from all layers
        with torch.no_grad():
            outs = self.target_model(cond_idx=cond_idx, input_ids=input_ids, attention_mask=attention_mask)
        
        # # Eagle 3: Extract hidden states from 3 different layers
        # # Use layer 0, 1, and 2 (early, middle, late features)
        hidden_states0 = outs.hidden_states[0]  # Early layer features
        hidden_states1 = outs.hidden_states[1]  # Middle layer features  
        hidden_states2 = outs.hidden_states[2]  # Late layer features
        
        # # Concatenate the 3 layer features
        hidden_states = torch.cat((hidden_states0, hidden_states1, hidden_states2), dim=-1)

        target = outs["logits"]
        target = padding(target, left=False) # pad by one token for 119
        input_ids = padding(input_ids, left=False)

        if target is not None:
            target = target.to(device)
            loss_mask = loss_mask[..., None]
            loss_mask = loss_mask.to(device)

        return hidden_states, target, loss_mask, input_ids

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            seq_length_with_past = input_shape[-1] + past_key_values_length
            if attention_mask.shape[1] < seq_length_with_past: # this part won't be needed.
                attention_mask = F.pad(attention_mask, (0, seq_length_with_past - attention_mask.shape[1]), "constant", True)
            
            expanded_attn_mask = _expand_mask(attention_mask, torch.float32, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask


    def forward(
            self,
            input_ids,
            cond_idx: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            loss_mask: Optional[torch.Tensor] = None,
    ):
        """
        Eagle 3 Multi-Step Training Forward Pass for LlamaGen (fixed for checkpointing + DDP)

        Returns:
            plosses: list of torch.Tensor (per-step losses, requires_grad=True)
            vlosses: list (left empty here, preserved for compatibility)
            acces: list of floats (per-step accuracies, detached)
        """
        # Eagle 3: Extract multi-layer features
        # input_ids's are padded with empty tokens during preprocessing of training data !!!
        hidden_states, target, loss_mask, input_ids = self.dataprepare(cond_idx, input_ids, attention_mask, loss_mask)

        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0


        # Eagle 3: Process concatenated 3-layer features
        hidden_states = self.fc(hidden_states)

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            device = hidden_states.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if self.freqs_cis.device != position_ids.device:
            self.freqs_cis = self.freqs_cis.to(position_ids.device)
        freqs_cis = self.freqs_cis[position_ids].squeeze(0)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )
      
        # Eagle 3: Multi-step iterative training
        plosses = []
        vlosses = []
        acces = []
        cache_hidden = [[], []]

        # precompute device and freqs
        device = hidden_states.device
        freqs_cis_full = self.freqs_cis.to(device)

        for idx in range(self.length):
            last = idx == self.length - 1
            # [SY] re-inject cond_idx dropped during preprocessing
            cond_idx_embedding = self.target_model.model.cls_embedding(cond_idx, train=self.target_model.model.training)[:, :120]
            inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = torch.cat((cond_idx_embedding , inputs_embeds), dim=1)

            inputs_embeds = inputs_embeds.to(hidden_states.dtype)

            layer_outputs, cache_hidden = self.midlayer(
                input_emb=inputs_embeds,
                hidden_states=hidden_states,
                cache_hidden=cache_hidden,                    
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=output_attentions,
                use_cache=True,
            )

            # hidden_states_out = layer_outputs[0]
            # hidden_states_out = layer_outputs # [SY] not returning a tuple of trivial stuff anymore !!
            hidden_states_out = layer_outputs[0]

            with torch.no_grad():
                # hidden_states_target = padding(hidden_states, left=False)
                target_head = target

                # [SY] disable hidden state/token mapping of eagle3 since dimensions are the same

                target_max_token = target_head.argmax(-1) 
                target_mask = torch.ones_like(target_max_token, dtype=torch.int) # [SY] add this since all tokens are valid due to d=t
                # # Move d2t to the same device as target_max_token
                # self.t2d = self.t2d.to(target_max_token.device) 
                # target_mask = self.t2d[target_max_token] # for each target token (ID from target_max_token), get the corresponding drafter token or mask.
                target_mask = target_mask[..., None].int()
                position_mask = target_mask * loss_mask
                # target_head = target_head[..., self.t2d]
                # target_head = target_head.float()

                # [SY]: Loss scaling for merging at inference
                n_target_logits = F.normalize(target_head, dim=2, eps=1e-6) .to(torch.float32)
                cosine_sim_matrix = torch.bmm(n_target_logits, n_target_logits.transpose(1, 2))
                B, L, _ = cosine_sim_matrix.shape
                thresh = 0.625
                # build upper-triangular mask (i < j)
                upper_mask = torch.triu(torch.ones((L, L), device=cosine_sim_matrix.device), diagonal=1)  # [L, L]
                upper_mask = upper_mask.unsqueeze(0).expand(B, L, L)  # broadcast to [B, L, L]

                 # combine with threshold condition
                logit_sim_mask = (upper_mask.bool() & (cosine_sim_matrix > thresh)).float()
                token_weight_mask = (logit_sim_mask.sum(dim=2, keepdim=True) > 0).float()  # [B, L, 1]

                weight = 2.0

                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()


            hidden_states = hidden_states_out

            hidden_states_out = self.norm(hidden_states_out)

            logits = self.lm_head(hidden_states_out)
            logits = logits.float()
            out_logp = nn.LogSoftmax(dim=2)(logits)
            plogp = target_p * out_logp

            weighted_mask = position_mask * (1 + token_weight_mask * (weight - 1))  # [B, L, 1]
            # [SY]: new loss for loss-scaling
            # loss = -torch.sum(position_mask * plogp, 2).mean() # original loss of eagle3 drafter
            loss = -torch.sum(weighted_mask * plogp, 2).mean()

            plosses.append(loss)
            with torch.no_grad():
                acces.append(((logits.argmax(-1) == target_p.argmax(-1)) * position_mask.squeeze(-1)).sum().item() / (
                        loss_mask.sum().item() + 1e-6))

            if not last:
                input_ids = padding(input_ids, left=False)
                target = padding(target, left=False)
                loss_mask = padding(loss_mask, left=False)

        return plosses, vlosses, acces

