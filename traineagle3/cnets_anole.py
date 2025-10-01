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
from models.configs.configuration_anole import ChameleonConfig
from models.configs.configs import EConfig    
from models.drafters.choices import *

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

# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Chameleon
class ChameleonRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        ChameleonRMSNorm is equivalent to T5LayerNorm
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

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class LlamaRotaryEmbedding(nn.Module):
    """
    Llama Rotary Positional Embedding Module.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        """
        Set the cosine and sine cache for positional embeddings.
        """
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )

    def forward(self, x, seq_len=None):
        """
        Forward pass of the LlamaRotaryEmbedding module.
        """
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """
    Apply rotary position embeddings to query and key tensors.
    """
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.LlamaMLP with Llama->Chameleon
class ChameleonMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    # Ignore copy
    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class ChameleonLayerNorm(nn.LayerNorm):
    """
    LayerNorm but computes stats only over the last dim because Chameleon applies gamma and beta
    from each shard separately to each head, instead of reducing.
    """

    def __init__(self, hidden_size, model_parallel_size, n_heads_per_mp, *args, **kwargs):
        if isinstance(hidden_size, int):
            hidden_size = (hidden_size,)
        super().__init__([model_parallel_size, *hidden_size], *args, **kwargs)
        self.normalized_shape = (hidden_size[-1],)
        self.n_heads_per_mp = n_heads_per_mp

    def repeat_param(self, param):
        return param.repeat_interleave(self.n_heads_per_mp, dim=0)

    def forward(self, hidden_states):
        hidden_states = F.layer_norm(hidden_states, self.normalized_shape, None, None, eps=1e-5)
        hidden_states = hidden_states * self.repeat_param(self.weight) + self.repeat_param(self.bias)
        return hidden_states

# Copied from transformers.models.llama.modeling_llama.repeat_kv
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


class ChameleonAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: ChameleonConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.model_parallel_size = config.model_parallel_size

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Eagle 3: Modified to accept concatenated input features
        self.q_proj = nn.Linear(self.hidden_size * 2, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.q_norm = ChameleonLayerNorm(
            self.head_dim, self.model_parallel_size, self.num_heads // self.model_parallel_size
        )
        self.k_norm = ChameleonLayerNorm(
            self.head_dim, self.model_parallel_size, self.num_key_value_heads // self.model_parallel_size
        )
        self._init_rope()

    def _init_rope(self):
        assert self.config.rope_scaling is None, "RoPE scaling is not supported in ChameleonAttention"
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(-1, self.num_heads, self.head_dim)
        query_states = self.q_norm(query_states)

        key_states = key_states.reshape(-1, self.num_key_value_heads, self.head_dim)
        key_states = self.k_norm(key_states)

        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
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


CHAMELEON_ATTENTION_CLASSES = {
    "eager": ChameleonAttention,
}


class ChameleonDecoderLayer(nn.Module):
    def __init__(self, config: ChameleonConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = CHAMELEON_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = ChameleonMLP(config)
        self.input_layernorm = ChameleonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = ChameleonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Eagle 3 Style Decoder Layer with input embedding fusion for Anole
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
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + self.dropout(hidden_states)
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

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

class Eagle3AnoleModel(nn.Module):
    """
    Eagle 3 Model for Anole with Multi-Layer Feature Extraction
    """
    def __init__(self, config, training_config, load_emb=False, path=None, bias=True, total_tokens=63, depth=5, top_k=8, threshold=1.0, embed_upscale=1.0):
        super().__init__()

        self.config = config
        self.training_config = training_config
        self.gradient_checkpointing = training_config.gradient_checkpointing if hasattr(training_config, 'gradient_checkpointing') else True
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.length = 7  # Eagle 3 multi-step training length

        # Load target model for multi-layer feature extraction
        if path is not None:
            # For Anole, use the appropriate model class
            from models.base_models.anole.modeling_anole import AnoleModel
            self.target_model = AnoleModel.from_pretrained(path, torch_dtype=torch.float16, output_hidden_states=True)
            self.target_model.eval()
            for param in self.target_model.parameters():
                param.requires_grad = False
        else:
            self.target_model = None

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        if load_emb:
            from safetensors import safe_open
            import json
            try:
                with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                with safe_open(os.path.join(path, emb_path),
                               framework="pt",
                               device="cpu") as f:
                    tensor_slice = f.get_slice("model.embed_tokens.weight")
                    vocab_size, hidden_dim = tensor_slice.get_shape()
                    tensor = tensor_slice[:, :hidden_dim].float()
            except:
                with open(os.path.join(path, "pytorch_model.bin.index.json"), "r") as f:
                    index_json = json.loads(f.read())
                    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
                weights = torch.load(os.path.join(path, emb_path))
                tensor = weights["model.embed_tokens.weight"].float()
            self.embed_tokens.weight.data = tensor

        self.top_k = top_k
        self.total_tokens = total_tokens - 1
        self.depth = depth
        self.threshold = math.log(threshold)
        self.embed_upscale = embed_upscale

        # Eagle 3: Single decoder layer for multi-step processing
        self.midlayer = ChameleonDecoderLayer(config, 0)
        
        # Eagle 3: 3-feature input fusion (hidden_states from 3 layers)
        self.fc = nn.Linear(3 * config.hidden_size, config.hidden_size, bias=False)
        
        self.norm = ChameleonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        
        # Anole specific: non-image tokens
        self.non_image_tokens = [i for i in range(0, 4)] + [i for i in range(8196, 65536)]
        self.non_image_tokens = torch.tensor(self.non_image_tokens)
        
        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def dataprepare(self, input_ids, attention_mask, loss_mask):
        """
        Eagle 3: Extract features from 3 different layers of the target model (Anole)
        """
        device = input_ids.device
        
        # Get outputs with hidden states from all layers
        with torch.no_grad():
            outs = self.target_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # Eagle 3: Extract hidden states from 3 different layers
        # Use layer 0, 1, and 2 (early, middle, late features)
        hidden_states0 = outs.hidden_states[0]  # Early layer features
        hidden_states1 = outs.hidden_states[1]  # Middle layer features  
        hidden_states2 = outs.hidden_states[2]  # Late layer features
        
        # Concatenate the 3 layer features
        hidden_states = torch.cat((hidden_states0, hidden_states1, hidden_states2), dim=-1)
        
        target = outs.logits
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
        Eagle 3 Multi-Step Training Forward Pass for Anole
        """
        # Eagle 3: Extract multi-layer features
        hidden_states, target, loss_mask, input_ids = self.dataprepare(input_ids, attention_mask, loss_mask)

        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if self.training and self.gradient_checkpointing and not hidden_states.requires_grad:
            hidden_states.requires_grad = True

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

        for idx in range(self.length):
            last = idx == self.length - 1
            inputs_embeds = self.embed_tokens(input_ids)
            if self.training and self.gradient_checkpointing and not inputs_embeds.requires_grad:
                inputs_embeds.requires_grad = True
            inputs_embeds = inputs_embeds.to(hidden_states.dtype)

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, None, output_attentions)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.midlayer),
                    inputs_embeds,
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
            else:
                layer_outputs = self.midlayer(
                    input_emb=inputs_embeds,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=output_attentions,
                    use_cache=True,
                )

            hidden_states_out = layer_outputs[0]

            with torch.no_grad():
                target_head = target
                target_max_token = target_head.argmax(-1)
                position_mask = loss_mask
                target_head = target_head.float()
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()

            hidden_states = hidden_states_out
            hidden_states_out = self.norm(hidden_states_out)

            # For Anole, use the full vocabulary size directly
            logits = torch.matmul(hidden_states_out, self.embed_tokens.weight.t())
            logits = logits.float()
            out_logp = nn.LogSoftmax(dim=2)(logits)
            plogp = target_p * out_logp
            loss = -torch.sum(position_mask * plogp, 2).mean()
            plosses.append(loss)
            
            with torch.no_grad():
                acces.append(((logits.argmax(-1) == target_p.argmax(-1)) * position_mask.squeeze(-1)).sum().item() / (
                        loss_mask.sum().item() + 1e-6))

            if not last:
                # Eagle 3: Progressive attention masking for next iteration
                input_ids = padding(input_ids, left=False)
                target = padding(target, left=False)
                loss_mask = padding(loss_mask, left=False)
                ind = torch.arange(seq_length, device=attention_mask.device)
                ind0 = ind[idx:]
                ind1 = ind[:seq_length-idx]
                attention_mask[:, :, ind0, ind1] = torch.finfo(attention_mask.dtype).min

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

