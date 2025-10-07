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
def precompute_freqs_cis_2d(grid_size, n_elem, rope_base=10000, cls_token_num=120):
    """
    Precompute frequencies for 2D RoPE as used in LlamaGen
    """
    # 1D RoPE - generates n_elem//2 complex frequencies for n_elem real dimensions
    def precompute_freqs_cis_1d(seq_len, n_elem, base=10000):
        freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2).float() / n_elem))
        t = torch.arange(seq_len, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    # 2D RoPE: For n_elem complex dimensions, we need n_elem/2 frequencies for each spatial direction
    # Each spatial direction gets n_elem/4 complex frequencies
    quarter_dim = n_elem // 4
    
    freqs_cis_h = precompute_freqs_cis_1d(grid_size, quarter_dim * 2, rope_base)  # generates quarter_dim complex numbers
    freqs_cis_w = precompute_freqs_cis_1d(grid_size, quarter_dim * 2, rope_base)  # generates quarter_dim complex numbers
    
    freqs_cis = torch.cat([
        freqs_cis_h[:, None, :].expand(grid_size, grid_size, quarter_dim).flatten(0, 1),
        freqs_cis_w[None, :, :].expand(grid_size, grid_size, quarter_dim).flatten(0, 1),
    ], dim=-1)

    # Cache should match the frequency tensor dimensions (n_elem/2 complex numbers)
    cache = torch.cat([
        torch.zeros(cls_token_num, n_elem // 2, dtype=torch.complex64),
        freqs_cis
    ], dim=0)

    return cache

def apply_rotary_emb(xq, xk, freqs_cis):
    """Apply rotary embeddings to query and key tensors.
    
    Args:
        xq: query states with shape [batch, num_heads, seq_len, head_dim] (after transpose)
        xk: key states with shape [batch, num_key_value_heads, seq_len, head_dim] (after transpose)  
        freqs_cis: frequency tensor with shape [seq_len, head_dim//2]
    """
    # Get sequence length from xq (dimension 2 after transpose)
    seq_len = xq.shape[2]
    
    # Handle case where sequence length exceeds freqs_cis length
    if seq_len > freqs_cis.shape[0]:
        # Extend freqs_cis by repeating the last few entries
        num_missing = seq_len - freqs_cis.shape[0]
        # Repeat the last 10 entries cyclically to fill the gap
        last_entries = freqs_cis[-10:]
        repeats = (num_missing + 9) // 10  # Ceiling division
        extended_freqs = last_entries.repeat(repeats, 1)[:num_missing]
        freqs_cis = torch.cat([freqs_cis, extended_freqs], dim=0)
    
    # Slice freqs_cis to match the sequence length
    freqs_cis = freqs_cis[:seq_len]  # [seq_len, head_dim//2]
    
    # Reshape for broadcasting: [1, 1, seq_len, head_dim//2]
    freqs_cis = freqs_cis[None, None, :, :]
    
    # Convert to complex and apply rotary embeddings
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

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

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
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

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        # Apply rotary embeddings using LlamaGen's method
        if freqs_cis is not None:
            query_states, key_states = apply_rotary_emb(query_states, key_states, freqs_cis)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        freqs_cis: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Eagle 3 Style Decoder Layer with input embedding fusion for LlamaGen
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        input_emb = self.input_layernorm(input_emb)  # Also normalize input embeddings

        # Eagle 3: Concatenate input embeddings with hidden states
        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            freqs_cis=freqs_cis,
        )
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

@torch.no_grad()
def padding(tensor, left=True):
    """Helper function for padding tensors"""
    zeropadding = torch.zeros_like(tensor[:, -1:])
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

class Eagle3LlamaGenModel(nn.Module):
    """
    Eagle 3 Model for LlamaGen with Multi-Layer Feature Extraction
    """
    def __init__(self, config, training_config, load_emb=False, path=None, bias=True, total_tokens=63, depth=5, top_k=8, threshold=1.0, embed_upscale=1.0):
        super().__init__()

        self.config = config
        self.training_config = training_config
        self.gradient_checkpointing = training_config.gradient_checkpointing if hasattr(training_config, 'gradient_checkpointing') else True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.length = 1  # Eagle 3 multi-step training length

        # Load target model for multi-layer feature extraction
        if path is not None:
            # [SY]: fix this 
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
        else:
            self.target_model = None
            self.target_hidden_size = config.hidden_size
            self.target_vocab_size = config.vocab_size

        # Initialize embedding layer with correct dimensions
        if load_emb and path is not None:
            # First, get the actual embedding dimensions from the target model
            from safetensors import safe_open
            import json
            embedding_dim = config.hidden_size  # Default fallback
            try:
                try:
                    head_path = "model.safetensors"
                    with safe_open(os.path.join(path, head_path),
                                framework="pt",
                                device="cpu") as f:
                        tensor_slice = f.get_slice("lm_head.weight")
                        vocab_size, embedding_dim = tensor_slice.get_shape()
                        tensor = tensor_slice[:, :embedding_dim].float()
                except:
                    head_path = "pytorch_model.bin"
                    weights = torch.load(os.path.join(path, head_path), weights_only=True)
                    tensor = weights["lm_head.weight"]
                    embedding_dim = tensor.shape[1]
            except Exception as e:
                print(f"Warning: Could not load embeddings: {e}")
                tensor = None
                embedding_dim = config.hidden_size
            
            # Create embedding layer with correct dimensions (use target vocab size)
            self.embed_tokens = nn.Embedding(self.target_vocab_size, embedding_dim, self.padding_idx)
            if tensor is not None:
                self.embed_tokens.weight.data = tensor
            # Add projection layer if dimensions don't match
            if embedding_dim != config.hidden_size:
                self.embed_proj = nn.Linear(embedding_dim, config.hidden_size, bias=False)
            else:
                self.embed_proj = None
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
            self.embed_proj = None
        
        # Add output head for proper logits computation (use target vocab size)
        if self.embed_proj is not None:
            # If we have embedding projection, we need output projection too
            self.output_head = nn.Linear(config.hidden_size, self.target_vocab_size, bias=False)
        else:
            self.output_head = None

        self.top_k = top_k
        self.total_tokens = total_tokens - 1
        self.depth = depth
        self.threshold = math.log(threshold)
        self.embed_upscale = embed_upscale

        # Eagle 3: Single decoder layer for multi-step processing
        self.midlayer = LlamaDecoderLayer(config, 0)
        
        # Eagle 3: 3-feature input fusion (hidden_states from 3 layers)
        # Use target model's hidden size instead of config hidden size
        self.fc = nn.Linear(3 * self.target_hidden_size, config.hidden_size, bias=False)
        
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        
        # LlamaGen specific configurations
        if hasattr(config, 'input_type'):
            if config.input_type == "c2i":
                config.block_size = 576
                config.rope_base = 10000
                config.cls_token_num = 0
            elif config.input_type == "t2i":
                config.block_size = 256
                config.rope_base = 10000
                config.cls_token_num = 119
            elif config.input_type == "t2i2":
                config.block_size = 1024
                config.rope_base = 10000
                config.cls_token_num = 119
        else:
            # Default values
            config.block_size = 256
            config.rope_base = 10000
            config.cls_token_num = 119
            
        grid_size = int(config.block_size ** 0.5)
        assert grid_size ** 2 == config.block_size, "block_size must be a perfect square"
        # Use target model config for correct head dimensions
        # For Eagle3, the head_dim should match the actual attention head dimension
        head_dim = config.hidden_size // config.num_attention_heads
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, head_dim, config.rope_base, config.cls_token_num)
        # expand the freqs_cis for few steps to avoid out of bound error
        padding_freqs = torch.zeros_like(self.freqs_cis[:10])
        self.freqs_cis = torch.cat([self.freqs_cis, padding_freqs], dim=0)
        
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def dataprepare(self, input_ids, attention_mask, loss_mask):
        """
        Eagle 3: Extract features from 3 different layers of the target model (LlamaGen)
        """
        device = input_ids.device
        
        # Get outputs with hidden states from all layers
        with torch.no_grad():
            outputs = self.target_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # # Eagle 3: Extract hidden states from 3 different layers
        # # Use layer 0, 1, and 2 (early, middle, late features)
        # hidden_states0 = outs.hidden_states[0]  # Early layer features
        # hidden_states1 = outs.hidden_states[1]  # Middle layer features  
        # hidden_states2 = outs.hidden_states[2]  # Late layer features
        
        # # Concatenate the 3 layer features
        # hidden_states = torch.cat((hidden_states0, hidden_states1, hidden_states2), dim=-1)

        if outputs["hidden_states"][0].device != device:
            outputs["hidden_states"] = [x.to(device) for x in outputs["hidden_states"]]
        hidden_states = torch.cat(outputs["hidden_states"], dim=-1)
        
        target = outputs["logits"]
        target = padding(target, left=False)
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
            if attention_mask.shape[1] < seq_length_with_past:
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
        hidden_states, target, loss_mask, input_ids = self.dataprepare(input_ids, attention_mask, loss_mask)

        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if self.training and self.gradient_checkpointing and not hidden_states.requires_grad:
            hidden_states.requires_grad_(True)

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

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        # Eagle 3: Multi-step iterative training
        plosses = []
        vlosses = []
        acces = []

        # precompute device and freqs
        device = hidden_states.device
        freqs_cis_full = self.freqs_cis.to(device)

        # prepare inputs_embeds once (we will mutate x inside the wrapper)
        inputs_embeds = self.embed_tokens(input_ids)
        if self.embed_proj is not None:
            inputs_embeds = self.embed_proj(inputs_embeds)
        if self.training and self.gradient_checkpointing and not inputs_embeds.requires_grad:
            inputs_embeds.requires_grad_(True)
        inputs_embeds = inputs_embeds.to(hidden_states.dtype)

        # Define the multistep function that will run the full iterative loop.
        # Keep losses as tensors so they can be backpropagated.
        def multistep_forward(
            input_ids,
            inputs_embeds,
            hidden_states,
            attention_mask,
            position_ids,
            target,
            loss_mask,
            output_attentions,
        ):
            ploss_list = []
            acc_list = []
            for idx in range(self.length):
                last = idx == self.length - 1

                # compute freqs_cis (same as before)
                if seq_length > self.freqs_cis.shape[0]:
                    num_missing = seq_length - self.freqs_cis.shape[0]
                    last_entries = self.freqs_cis[-10:]
                    repeats = (num_missing + 9) // 10
                    extended_freqs = last_entries.repeat(repeats, 1)[:num_missing]
                    extended_freqs_cis = torch.cat([self.freqs_cis, extended_freqs], dim=0)
                    freqs_cis = extended_freqs_cis[:seq_length].to(hidden_states.device)
                else:
                    freqs_cis = self.freqs_cis[:seq_length].to(hidden_states.device)

                layer_outputs = self.midlayer(
                    input_emb=inputs_embeds,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=output_attentions,
                    use_cache=True,
                    freqs_cis=freqs_cis,
                )
                hidden_states = layer_outputs[0]
                hidden_states_out = self.norm(hidden_states)
                if self.output_head is not None:
                    logits = self.output_head(hidden_states_out)
                else:
                    logits = hidden_states_out @ self.embed_tokens.weight.T

                out_logp = nn.LogSoftmax(dim=2)(logits)  # no casting

                with torch.no_grad():
                    target_head = target
                    target_p = nn.Softmax(dim=2)(target_head.float()).detach()
                plogp = target_p * out_logp
                loss = -torch.sum(loss_mask * plogp, 2).mean()
                ploss_list.append(loss)  # tensor, not float
                with torch.no_grad():
                    acc_list.append(((logits.argmax(-1) == target_p.argmax(-1)) * loss_mask.squeeze(-1)).sum().item() / (loss_mask.sum().item() + 1e-6))

                if not last:
                    input_ids = padding(input_ids, left=False)
                    target = padding(target, left=False)
                    loss_mask = padding(loss_mask, left=False)
                    ind = torch.arange(seq_length, device=attention_mask.device)
                    ind0 = ind[idx:]
                    ind1 = ind[:seq_length-idx]
                    attention_mask = attention_mask.clone()
                    attention_mask[:, :, ind0, ind1] = torch.finfo(attention_mask.dtype).min

            plosses_tensor = torch.stack(ploss_list)  # always 1D tensor
            acces_tensor = torch.tensor(acc_list, device=plosses_tensor.device)  # shape (length_int,)

            return plosses_tensor, acces_tensor

        # Run checkpointed whole-multistep (so midlayer params participate in a single reentrant backward)
        if self.gradient_checkpointing and self.training:
            # Pass Python ints for seq_length and length
            plosses_tensor, acces_tensor = checkpoint(
                multistep_forward,
                input_ids,
                inputs_embeds,
                hidden_states,
                attention_mask,
                position_ids,
                target,
                loss_mask,
                output_attentions,
            )
            
            # Keep plosses as tensors (do not convert to .item())
            plosses = [pl for pl in plosses_tensor.unbind(0)]
            # acces for logging only: convert to python floats
            acces = [float(ac.detach().cpu()) for ac in acces_tensor.unbind(0)]
        else:
            # fallback: non-checkpointed loop (keeps original semantics)
            for idx in range(self.length):
                last = idx == self.length - 1
                
                # Recompute inputs_embeds for each step in non-checkpointed mode
                current_inputs_embeds = self.embed_tokens(input_ids)
                if self.embed_proj is not None:
                    current_inputs_embeds = self.embed_proj(current_inputs_embeds)
                current_inputs_embeds = current_inputs_embeds.to(hidden_states.dtype)

                # compute freqs_cis (same as before)
                if seq_length > self.freqs_cis.shape[0]:
                    num_missing = seq_length - self.freqs_cis.shape[0]
                    last_entries = self.freqs_cis[-10:]
                    repeats = (num_missing + 9) // 10
                    extended_freqs = last_entries.repeat(repeats, 1)[:num_missing]
                    extended_freqs_cis = torch.cat([self.freqs_cis, extended_freqs], dim=0)
                    freqs_cis = extended_freqs_cis[:seq_length].to(hidden_states.device)
                else:
                    freqs_cis = self.freqs_cis[:seq_length].to(hidden_states.device)

                layer_outputs = self.midlayer(
                    input_emb=current_inputs_embeds,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=output_attentions,
                    use_cache=True,
                    freqs_cis=freqs_cis,
                )
                hidden_states = layer_outputs[0]
                hidden_states_out = self.norm(hidden_states)
                if self.output_head is not None:
                    logits = self.output_head(hidden_states_out)
                else:
                    logits = torch.matmul(hidden_states_out, self.embed_tokens.weight.t())
                logits = logits.float()
                out_logp = nn.LogSoftmax(dim=2)(logits)
                with torch.no_grad():
                    target_head = target
                    target_p = nn.Softmax(dim=2)(target_head.float()).detach()
                plogp = target_p * out_logp
                loss = -torch.sum(loss_mask * plogp, 2).mean()
                plosses.append(loss)  # tensor, not float
                with torch.no_grad():
                    acces.append(((logits.argmax(-1) == target_p.argmax(-1)) * loss_mask.squeeze(-1)).sum().item() / (loss_mask.sum().item() + 1e-6))

                if not last:
                    input_ids = padding(input_ids, left=False)
                    target = padding(target, left=False)
                    loss_mask = padding(loss_mask, left=False)
                    ind = torch.arange(seq_length, device=attention_mask.device)
                    ind0 = ind[idx:]
                    ind1 = ind[:seq_length-idx]
                    attention_mask = attention_mask.clone()
                    attention_mask[:, :, ind0, ind1] = torch.finfo(attention_mask.dtype).min

        # Return per-step losses (as tensors), placeholder vlosses, and per-step accuracies (floats)
        return plosses, vlosses, acces

 
    def init_tree(self, tree=None):
        if tree is not None:
            # EAGLE v1
            self.tree = tree
            self.tree_buffer=generate_tree_buffers(self.tree, self.embed_tokens.weight.device)
        else:
            # EAGLE v2
            self.tree_mask_init = torch.eye(self.top_k, device=self.embed_tokens.weight.device)[None, None]
            self.position_ids = torch.zeros(self.top_k, device=self.embed_tokens.weight.device, dtype=torch.long)
            self.tree_mask_init = self.tree_mask_init.to(self.embed_tokens.weight.device)

    def reset(self):
        self.tree_mask = None
