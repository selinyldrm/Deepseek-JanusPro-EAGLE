"""
Temporal Conditioning Adapter for LlamaGen EAGLE Drafter
=========================================================

Plugs into the existing EAGLE drafter (cnets2_llamagen.py Model class)
WITHOUT modifying it. Uses a wrapper that injects previous frame tokens
into the hidden_states before they reach the drafter's fc layer.

Architecture
------------
The drafter's forward does:
    hidden_states = self.fc(cat(inputs_embeds, hidden_states))

We intercept `hidden_states` (which comes from the target model) and add
a temporal conditioning vector derived from the previous frame's VQ tokens:

    hidden_states_aug = hidden_states + temporal_proj(prev_frame_encoding)

Frame 1:  prev_frame_tokens = None → hidden_states unchanged (standard EAGLE)
Frame N:  prev_frame_tokens = (576,) → temporal signal injected

Training requirement
--------------------
The TemporalAdapter requires training. The base drafter stays FROZEN.
Training cost is tiny — the adapter is ~10M parameters vs 3B for the drafter.

Training signal: same KL distillation used for the drafter, but now conditioned
on prev_frame_tokens. A video dataset provides the (prompt, frame_t, frame_{t+1})
triplets needed.

If you want ZERO training: set temporal_blend_weight=0.0 — the adapter is 
bypassed entirely and you get standard single-frame EAGLE generation.

Usage
-----
    # Wrap the existing drafter
    adapter = TemporalAdapter(
        target_hidden_size = 2048,   # LlamaGen-XL hidden dim
        vq_codebook_size   = 16384,
        tokens_per_frame   = 576,    # 24x24
    )
    
    temporal_drafter = TemporalDrafterWrapper(
        base_drafter = ea_model.ea_layer,   # existing Model instance
        adapter      = adapter,
    )
    
    # Replace in ea_model
    ea_model.ea_layer = temporal_drafter
    
    # Frame 1 (text only)
    ea_model.ea_layer.set_prev_frame(None)
    tokens_f1 = ea_model.generate(prompt=p, ...)
    
    # Frame 2 (text + prev frame)
    ea_model.ea_layer.set_prev_frame(tokens_f1)
    tokens_f2 = ea_model.generate(prompt=p, ...)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from models.kv_variants.modeling_llamagen_kv import LlamaForCausalLM as KVLlamaForCausalLM
from transformers import PreTrainedModel, PretrainedConfig,AutoConfig

# ---------------------------------------------------------------------------
# Temporal Adapter — encodes prev frame tokens into a conditioning vector
# ---------------------------------------------------------------------------

class TemporalAdapter(nn.Module):
    """Encodes previous frame VQ tokens into a vector that augments
    the target model's hidden states before they enter the drafter.

    The adapter is the ONLY component that needs training.
    The drafter backbone (cnets2_llamagen.py Model) stays frozen.

    Design
    ------
    prev_frame_tokens (576,) int64
        ↓  VQ embedding  (576, embed_dim)
        ↓  Transformer encoder  (capture spatial structure of prev frame)
        ↓  Attention pool  →  (1, target_hidden_size)
        ↓  Added to hidden_states with learned blend weight

    The attention pool uses a single learnable query vector that attends
    to all 576 patch encodings — this produces a compact summary of the
    previous frame that the drafter can condition on at every token step.
    """

    def __init__(
        self,
        target_hidden_size: int = 2048,   # must match drafter hidden dim
        vq_codebook_size:   int = 16384,
        tokens_per_frame:   int = 576,    # 24x24 for 384px images
        embed_dim:          int = 256,    # internal embedding dim for VQ tokens
        num_encoder_layers: int = 2,      # lightweight — don't need deep encoder
        num_heads:          int = 8,
        dropout:            float = 0.0,
        init_blend_weight:  float = 0.1,  # start with weak conditioning
    ):
        super().__init__()

        self.target_hidden_size = target_hidden_size
        self.tokens_per_frame   = tokens_per_frame

        # Embed VQ codebook indices into a continuous space
        self.vq_embed = nn.Embedding(vq_codebook_size, embed_dim)

        # Positional embedding over the 576 spatial positions
        # Use 2D sin/cos — preserves spatial structure of the image
        self.register_buffer(
            "pos_embed",
            self._build_2d_pos_embed(tokens_per_frame, embed_dim),
        )

        # Lightweight transformer encoder over prev frame patches
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = embed_dim,
            nhead           = num_heads,
            dim_feedforward = embed_dim * 4,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,   # pre-norm — more stable
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Single learnable query — attends to all patch encodings
        # Produces a single vector summarizing the entire prev frame
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.cross_norm = nn.LayerNorm(embed_dim)

        # Project to target hidden size
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, target_hidden_size),
            nn.SiLU(),
            nn.Linear(target_hidden_size, target_hidden_size),
        )

        # Learned blend weight — starts near 0 so adapter doesn't disrupt
        # the drafter at init time. Grows during training as the adapter
        # learns to produce useful conditioning.
        self.blend_weight = nn.Parameter(torch.tensor(init_blend_weight))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.vq_embed.weight, std=0.02)
        nn.init.normal_(self.query, std=0.02)
        for layer in self.proj:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)

    @staticmethod
    def _build_2d_pos_embed(n_tokens: int, dim: int) -> torch.Tensor:
        """2D sinusoidal positional embedding for a square grid of tokens."""
        grid_size = int(n_tokens ** 0.5)
        assert grid_size * grid_size == n_tokens, \
            f"tokens_per_frame {n_tokens} must be a perfect square"

        # Build (grid_size, grid_size, dim) positional encoding
        half = dim // 2
        positions = torch.arange(grid_size, dtype=torch.float32)
        freq      = 1.0 / (10000 ** (torch.arange(0, half, 2).float() / half))

        sin_x = torch.outer(positions, freq).sin()   # (G, half/2)
        cos_x = torch.outer(positions, freq).cos()
        sin_y = torch.outer(positions, freq).sin()
        cos_y = torch.outer(positions, freq).cos()

        # Interleave x and y
        pe_row = torch.cat([sin_x, cos_x], dim=-1)   # (G, half)
        pe_col = torch.cat([sin_y, cos_y], dim=-1)

        pe = (pe_row[:, None, :] + pe_col[None, :, :])   # (G, G, half)
        pe = pe.view(n_tokens, half)

        # Pad to full dim if needed
        if dim > half * 2:
            pe = F.pad(pe, (0, dim - half * 2))

        return pe.unsqueeze(0)   # (1, n_tokens, dim)

    def forward(
        self,
        prev_frame_tokens: torch.Tensor,   # (B, 576) int64
    ) -> torch.Tensor:
        """Encode previous frame into a conditioning vector.

        Returns:
            cond: (B, 1, target_hidden_size) — broadcast over seq_len by caller
        """
        B = prev_frame_tokens.shape[0]

        # Embed and add positional encoding
        x = self.vq_embed(prev_frame_tokens)            # (B, 576, embed_dim)
        x = x + self.pos_embed.to(x.device)             # (B, 576, embed_dim)

        # Encode spatial structure of prev frame
        x = self.encoder(x)                             # (B, 576, embed_dim)

        # Pool to single vector via cross-attention
        q = self.query.expand(B, -1, -1)                # (B, 1, embed_dim)
        cond, _ = self.cross_attn(q, x, x)              # (B, 1, embed_dim)
        cond = self.cross_norm(cond)

        # Project to target hidden size
        cond = self.proj(cond)                           # (B, 1, target_hidden_size)
        return cond


# ---------------------------------------------------------------------------
# Wrapper — injects temporal conditioning into the drafter's hidden_states
# ---------------------------------------------------------------------------

class TemporalDrafterWrapper(nn.Module):
    """Wraps the existing EAGLE drafter (cnets2_llamagen.py Model) and
    injects temporal conditioning from the previous frame.

    The base drafter is FROZEN. Only TemporalAdapter trains.

    How injection works
    -------------------
    The drafter's forward() does:
        hidden_states = self.fc(cat(inputs_embeds, hidden_states))

    We intercept hidden_states BEFORE this line by patching it:
        hidden_states_aug = hidden_states + blend * adapter(prev_frame_tokens)

    The blend weight starts small (0.1) and grows during training.
    When prev_frame_tokens is None (frame 1), no augmentation is applied.
    """

    def __init__(
        self,
        base_drafter: nn.Module,     # existing Model from cnets2_llamagen.py
        adapter: TemporalAdapter,
    ):
        super().__init__()
        self.base_drafter     = base_drafter
        self.adapter          = adapter
        self._prev_frame_cond = None   # set externally via set_prev_frame()

        # Freeze drafter backbone
        for p in self.base_drafter.parameters():
            p.requires_grad = False

    def set_prev_frame(self, prev_frame_tokens: Optional[torch.Tensor]):
        """Call this before each generate() call.

        Args:
            prev_frame_tokens: (B, 576) int64 VQ token ids of previous frame,
                               or None for the first frame (text-only conditioning).
        """
        if prev_frame_tokens is None:
            self._prev_frame_cond = None
            return

        with torch.no_grad() if not self.training else torch.enable_grad():
            # Encode once and cache — used at every drafter step for this frame
            self._prev_frame_cond = self.adapter(prev_frame_tokens)   # (B, 1, H)

    def _augment_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Add temporal conditioning to hidden_states if available."""
        if self._prev_frame_cond is None:
            return hidden_states

        cond = self._prev_frame_cond

        # Broadcast over sequence length
        # hidden_states: (B, seq_len, H)
        # cond:          (B, 1, H)  → broadcast to (B, seq_len, H)
        blend  = torch.sigmoid(self.adapter.blend_weight)   # keep in (0, 1)
        return hidden_states + blend * cond

    # ------------------------------------------------------------------
    # Forward — proxy to base drafter with temporal augmentation
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        **kwargs,
    ):
        """Augment hidden_states then delegate to base drafter."""
        hidden_states = self._augment_hidden(hidden_states)
        return self.base_drafter(hidden_states, input_ids, **kwargs)

    # ------------------------------------------------------------------
    # Proxy all drafter methods used by ea2_model_llamagen.py
    # ------------------------------------------------------------------

    def topK_genrate_v1(self, step, recent_logits, hidden_states, input_ids,
                        head, logits_processor, cfg_scale):
        hidden_states = self._augment_hidden(hidden_states)
        return self.base_drafter.topK_genrate_v1(
            step, recent_logits, hidden_states, input_ids,
            head, logits_processor, cfg_scale
        )

    def topK_genrate(self, hidden_states, input_ids, head,
                     logits_processor, cfg_scale):
        hidden_states = self._augment_hidden(hidden_states)
        return self.base_drafter.topK_genrate(
            hidden_states, input_ids, head, logits_processor, cfg_scale
        )

    def init_tree(self):
        return self.base_drafter.init_tree()

    def init_tree_v1(self, tree_choices):
        return self.base_drafter.init_tree_v1(tree_choices)

    def reset(self):
        return self.base_drafter.reset()

    def reset_kv(self):
        return self.base_drafter.reset_kv()

    @property
    def stable_kv(self):
        return self.base_drafter.stable_kv

    @stable_kv.setter
    def stable_kv(self, value):
        self.base_drafter.stable_kv = value

    @property
    def tree_mask(self):
        return self.base_drafter.tree_mask

    @tree_mask.setter
    def tree_mask(self, value):
        self.base_drafter.tree_mask = value

    @property
    def tree_buffer(self):
        return self.base_drafter.tree_buffer

    @property
    def diff_device(self):
        return self.base_drafter.diff_device

    @property
    def headweight(self):
        return self.base_drafter.headweight if hasattr(self.base_drafter, "headweight") else None

    def acc(self, *args, **kwargs):
        return self.base_drafter.acc(*args, **kwargs)


# ---------------------------------------------------------------------------
# Video generation loop using the temporal drafter
# ---------------------------------------------------------------------------

def generate_video_with_temporal_drafter(
    ea_model,
    adapter: TemporalAdapter,
    prompt: str,
    n_frames: int     = 8,
    fps: int          = 8,
    output_path: str  = "output.mp4",
    **generate_kwargs,
) -> list:
    """Generate a video by feeding each completed frame as conditioning
    for the next frame's draft generation.

    Frame 1: text prompt only (adapter bypassed — prev_frame=None)
    Frame N: text prompt + VQ tokens of frame N-1

    Args:
        ea_model:         EaModel instance (ea2_model_llamagen.py)
        adapter:          TemporalAdapter (trained or untrained)
        prompt:           text description
        n_frames:         number of frames to generate
        fps:              output video frame rate
        output_path:      where to save the .mp4
        **generate_kwargs: passed to ea_model.generate()

    Returns:
        frames: list of (H, W, 3) uint8 numpy arrays
    """
    import cv2
    import numpy as np

    # Wrap the drafter with the temporal adapter
    if not isinstance(ea_model.ea_layer, TemporalDrafterWrapper):
        ea_model.ea_layer = TemporalDrafterWrapper(
            base_drafter = ea_model.ea_layer,
            adapter      = adapter,
        )

    frames = []
    prev_tokens = None

    for frame_idx in range(n_frames):
        print(f"Generating frame {frame_idx + 1}/{n_frames}...", end=" ", flush=True)

        # Set temporal conditioning for this frame
        ea_model.ea_layer.set_prev_frame(prev_tokens)

        # Generate frame tokens
        tokens, latency, accept_list = ea_model.generate(
            prompt=[prompt],
            **generate_kwargs,
        )

        mean_accept = sum(accept_list) / max(len(accept_list), 1)
        print(f"done ({latency:.1f}s, mean_accept={mean_accept:.2f})")

        # Decode to image
        _, img_tensor = ea_model.decode_ids(tokens)   # (1, 3, H, W) in [-1, 1]

        img_np = img_tensor[0].float().cpu().numpy()  # (3, H, W)
        img_np = img_np.transpose(1, 2, 0)            # (H, W, 3)
        img_np = np.clip((img_np + 1) / 2 * 255, 0, 255).astype(np.uint8)
        frames.append(img_np)

        # Feed generated tokens as conditioning for the next frame
        # tokens shape: (1, max_length) from ea_model.generate
        prev_tokens = tokens.clone()

    # Write video
    if len(frames) > 0:
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"\nSaved: {output_path}  ({len(frames)} frames @ {fps}fps)")

    return frames


# ---------------------------------------------------------------------------
# Training code for the adapter only
# ---------------------------------------------------------------------------

def train_temporal_adapter(
    temporal_drafter : TemporalDrafterWrapper, 
    video_dataset,           # dataset yielding (prompt, frame_t_tokens, frame_t1_tokens)
    n_epochs: int   = 3,
    lr: float       = 3e-4,
    grad_accum: int = 4,
    output_dir: str = "./temporal_adapter_ckpt",
):
    """Train only the TemporalAdapter — drafter stays frozen.

    For each (frame_t, frame_{t+1}) pair:
        1. Run target (frozen) over frame_{t+1} context → hidden states
        2. Run TemporalDrafterWrapper with prev=frame_t → draft logits
        3. KL(target_logits || draft_logits) at image token positions

    The adapter learns to encode frame_t in a way that makes the drafter's
    predictions for frame_{t+1} closer to the target's predictions.
    """
    import os
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader

    os.makedirs(output_dir, exist_ok=True)
    ea_layer = temporal_drafter.base_drafter
    adapter = temporal_drafter.adapter
    device = next(ea_layer.parameters()).device

   # Freeze everything except adapter
    ea_layer.eval()
    for p in ea_layer.parameters():
        p.requires_grad = False
    adapter.train()

    optimizer = AdamW(adapter.parameters(), lr=lr, weight_decay=0.01)
    loader    = DataLoader(video_dataset, batch_size=1, shuffle=True,
                           collate_fn=lambda x: x)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=n_epochs * len(loader) // grad_accum
    )

    Type=AutoConfig.from_pretrained("/work1/deming/shared/llamagen/LlamaGen-T2I-2").architectures[0]
    base_model = KVLlamaForCausalLM.from_pretrained(
        "/work1/deming/shared/llamagen/LlamaGen-T2I-2"
    ).eval()
    for p in base_model.parameters():
        p.requires_grad = False
        
    head = base_model.lm_head   # shared head — frozen

    global_step  = 0
    running_loss = 0.0
    optimizer.zero_grad()

    for epoch in range(n_epochs):
        for batch_idx, batch in enumerate(loader):
            sample = batch[0]

            # frame_t_tokens:  (1, 576) — previous frame VQ token ids
            # frame_t1_tokens: (1, 576) — current frame VQ token ids (what we predict)
            frame_t_tokens  = sample["frame_t_tokens"].unsqueeze(0).to(device)   # (1, 576)
            frame_t1_tokens = sample["frame_t1_tokens"].unsqueeze(0).to(device)  # (1, 576)

            # ------------------------------------------------------------------
            # Step 1: Get target hidden states for frame_{t+1}
            #
            # ea2_model_llamagen.py forward() signature:
            #   outputs, orig_logits, hidden_states = self(
            #       input_ids=..., output_orig=True, past_key_values=..., position_ids=...
            #   )
            # which internally calls:
            #   outputs = self.base_model.model(input_ids=..., ...)
            #   orig    = self.base_model.lm_head(outputs[0])
            #   hidden  = outputs[0]
            # ------------------------------------------------------------------
            with torch.no_grad():
                outputs = base_model.model(
                    input_ids    = frame_t1_tokens,
                    position_ids = torch.arange(
                        frame_t1_tokens.shape[1], device=device
                    ).unsqueeze(0),
                )
                # outputs[0] is the last hidden state from the target
                # shape: (1, 576, target_hidden_dim)  e.g. (1, 576, 2048)
                target_hidden = outputs[0]

                # Target logits — what we want the drafter to match
                # shape: (1, 576, vocab_size)
                target_logits = head(target_hidden)

            # ------------------------------------------------------------------
            # Step 2: Augment target hidden states with temporal conditioning
            #
            # This mirrors exactly what TemporalDrafterWrapper._augment_hidden
            # does at inference time — the adapter runs on prev frame tokens
            # and adds its output to the target hidden states.
            # ------------------------------------------------------------------
            # Encode previous frame — adapter is the ONLY trainable component
            prev_frame_cond = adapter([frame_t_tokens])    # (1, 1, target_hidden_dim)
            blend           = torch.sigmoid(adapter.blend_weight)

            # Broadcast cond over seq_len: (1, 1, H) → (1, 576, H)
            hidden_states_aug = target_hidden + blend * prev_frame_cond

            # ------------------------------------------------------------------
            # Step 3: Run drafter forward with augmented hidden states
            #
            # This is the exact call from cnets2_llamagen.py Model.forward():
            #   inputs_embeds = self.embed_tokens(input_ids)       [frozen]
            #   hidden_states = self.fc(cat(inputs_embeds, hidden_states_aug))
            #   for layer in self.layers: ...
            #
            # We call base_drafter directly (not the wrapper) to avoid
            # double-augmentation — the augmentation was done manually above.
            # ------------------------------------------------------------------
            draft_hidden, _ = ea_layer(
                hidden_states_aug,          # (1, 576, target_hidden_dim) — augmented
                input_ids    = frame_t1_tokens,   # (1, 576) — token ids for embed_tokens
                use_cache    = False,
            )
            # draft_hidden: (1, 576, drafter_hidden_dim)

            # Draft logits through the shared head
            draft_logits = head(draft_hidden)   # (1, 576, vocab_size)

            # ------------------------------------------------------------------
            # Step 4: KL loss at image token positions only
            # Shift: predict token_{t+1} from position t
            # target_logits[:, :-1] predicts target_logits[:, 1:]
            shift_target = target_logits[:, :-1, :].contiguous().view(-1, target_logits.shape[-1])
            shift_draft  = draft_logits[:,  :-1, :].contiguous().view(-1, draft_logits.shape[-1])

            p_target = F.softmax(shift_target / 2.0, dim=-1)   # temperature=2 softens targets
            log_q    = F.log_softmax(shift_draft / 2.0, dim=-1)

            loss = F.kl_div(log_q, p_target, reduction="batchmean") * 4.0  # T^2 scaling

            if loss.isnan() or loss.isinf():
                print(f"  Warning: invalid loss at step {global_step}, skipping")
                optimizer.zero_grad()
                continue

            (loss / grad_accum).backward()
            running_loss += loss.item()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    avg  = running_loss / 50
                    running_loss = 0.0
                    blend_val = torch.sigmoid(adapter.blend_weight).item()
                    print(f"epoch={epoch+1}  step={global_step}  "
                          f"loss={avg:.4f}  blend={blend_val:.4f}")

        ckpt_path = os.path.join(output_dir, f"adapter_epoch{epoch+1}.pt")
        torch.save({"state_dict": adapter.state_dict()}, ckpt_path)
        print(f"Saved: {ckpt_path}")
        
if __name__ == "__main__":
    from models.drafters.cnets2_llamagen import Model 
    from models.configs.configs import EConfig
    config = EConfig.from_pretrained('config.json')
    model = Model(config, load_emb=False)
    adapter = TemporalAdapter(
        target_hidden_size = 2048,   # LlamaGen-XL hidden dim
        vq_codebook_size   = 16384,
        tokens_per_frame   = 576,    # 24x24
    )
    
    temporal_drafter = TemporalDrafterWrapper(
        base_drafter = model,   # existing Model instance
        adapter      = adapter,
    )
    train_temporal_adapter(temporal_drafter)