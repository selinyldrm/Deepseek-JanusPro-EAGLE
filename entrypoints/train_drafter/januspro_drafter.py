"""
EAGLE-style Speculative Drafter Training for Janus-Pro
=======================================================

Architecture
------------
The drafter is a small transformer that receives:
  - The target model's last hidden state at position t   (4096-dim for Janus-Pro-7B)
  - The embedding of the token at position t
and predicts the image token at position t+1.

This is the EAGLE formulation: the drafter does NOT run a full LLM forward pass.
It runs one cheap transformer block conditioned on the target's hidden state,
which already encodes the full causal context including text conditioning.

Training objective
------------------
For each image generation sequence:
  1. Run target (frozen) to collect hidden states H = [h_1, ..., h_T]
  2. For each position t, train drafter to predict token_{t+1}
     given (h_t, embed(token_t))
  3. Loss = CrossEntropy over image token positions only

Data
----
Any dataset of text-image pairs works. The script uses a simple folder of
images with corresponding .txt prompt files. Replace the dataset class
with your own data loader as needed.

Usage
-----
  torchrun --nproc_per_node=8 janus_drafter_train.py \
      --target_model deepseek-ai/Janus-Pro-7B \
      --data_dir /path/to/images \
      --output_dir ./drafter_ckpt \
      --batch_size 4 \
      --lr 3e-4 \
      --epochs 3

Requirements
------------
  pip install transformers accelerate torch torchvision pillow einops
  pip install git+https://github.com/deepseek-ai/Janus.git
"""

from __future__ import annotations

import argparse
import math
import os
import glob
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import AutoModelForCausalLM, AutoConfig
from PIL import Image

# Janus-Pro imports
from janus.models import MultiModalityCausalLM, VLChatProcessor


# ---------------------------------------------------------------------------
# Drafter model
# ---------------------------------------------------------------------------

class DrafterConfig:
    """Configuration for the EAGLE-style drafter."""
    def __init__(
        self,
        hidden_size: int = 4096,       # must match target model hidden dim
        drafter_dim: int = 1024,       # drafter internal dimension — smaller = faster
        num_heads: int = 16,
        num_layers: int = 1,           # EAGLE uses 1 layer — fast and sufficient
        vocab_size: int = 116399,      # full Janus-Pro vocab size
        image_token_start: int = 100016,  # first image token index in Janus-Pro vocab
        image_token_end: int = 100593,    # last image token index
        max_seq_len: int = 576,        # 24x24 image tokens for 384px images
        dropout: float = 0.0,
    ):
        self.hidden_size       = hidden_size
        self.drafter_dim       = drafter_dim
        self.num_heads         = num_heads
        self.num_layers        = num_layers
        self.vocab_size        = vocab_size
        self.image_token_start = image_token_start
        self.image_token_end   = image_token_end
        self.max_seq_len       = max_seq_len
        self.dropout           = dropout


class DrafterBlock(nn.Module):
    """Single transformer block for the drafter — pre-norm, no cross-attention."""
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with causal mask
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, attn_mask=attn_mask, is_causal=True)
        x = x + residual

        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = self.ff(x) + residual
        return x


class JanusDrafter(nn.Module):
    """EAGLE-style drafter for Janus-Pro image generation.

    Takes target hidden states + token embeddings and predicts next image token.

    Forward pass (training):
        hidden_states: (B, T, H)  — target hidden states at each image position
        input_ids:     (B, T)     — image token ids at each position
        labels:        (B, T)     — next token ids (shifted by 1)

    Forward pass (inference / speculative step):
        hidden_state:  (B, 1, H) — target hidden state at current position
        input_id:      (B, 1)    — current image token id
        past_kv:       cached key-values from previous steps
    """
    def __init__(self, cfg: DrafterConfig):
        super().__init__()
        self.cfg = cfg

        # Project target hidden state into drafter dimension
        self.hidden_proj = nn.Linear(cfg.hidden_size, cfg.drafter_dim, bias=False)

        # Token embedding — shared vocabulary with target
        # Only embed image tokens for efficiency; offset by image_token_start
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.drafter_dim)

        # Positional embedding over image sequence
        self.pos_emb = nn.Embedding(cfg.max_seq_len + 1, cfg.drafter_dim)

        # Fusion: combine projected hidden state + token embedding
        # Simple learned gate: output = sigmoid(gate) * h_proj + (1 - gate) * tok_emb
        self.fusion_gate = nn.Linear(cfg.drafter_dim * 2, cfg.drafter_dim, bias=True)

        # Drafter transformer layers
        self.layers = nn.ModuleList([
            DrafterBlock(cfg.drafter_dim, cfg.num_heads, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])

        self.norm = nn.LayerNorm(cfg.drafter_dim)

        # Output head — project to full vocab, mask non-image tokens at loss time
        self.head = nn.Linear(cfg.drafter_dim, cfg.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,    # (B, T, H) target hidden states
        input_ids: torch.Tensor,        # (B, T) image token ids
        labels: Optional[torch.Tensor] = None,   # (B, T) for training
    ) -> dict:
        B, T, _ = hidden_states.shape
        device   = hidden_states.device

        # Project target hidden states
        h = self.hidden_proj(hidden_states)            # (B, T, D)

        # Token embeddings
        tok = self.tok_emb(input_ids)                  # (B, T, D)

        # Positional embeddings
        pos = torch.arange(T, device=device).unsqueeze(0)   # (1, T)
        pos_emb = self.pos_emb(pos)                    # (1, T, D)

        # Fuse hidden state + token embedding via gated sum
        fusion_input = torch.cat([h, tok], dim=-1)     # (B, T, 2D)
        gate   = torch.sigmoid(self.fusion_gate(fusion_input))  # (B, T, D)
        x      = gate * h + (1.0 - gate) * tok         # (B, T, D)
        x      = x + pos_emb

        # Causal self-attention through drafter layers
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.head(x)                          # (B, T, V)

        if labels is None:
            return {"logits": logits}

        # Loss — only on image token positions
        # Shift: predict token_{t+1} from state at t
        shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
        shift_labels = labels[:, 1:].contiguous()       # (B, T-1)

        # Mask: only compute loss on valid image token label positions
        image_mask = (
            (shift_labels >= self.cfg.image_token_start) &
            (shift_labels <  self.cfg.image_token_end)
        )

        if image_mask.sum() == 0:
            loss = shift_logits.sum() * 0.0   # no valid positions — zero loss, keep graph
        else:
            loss = F.cross_entropy(
                shift_logits[image_mask],
                shift_labels[image_mask],
                reduction="mean",
            )

        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def draft_one(
        self,
        hidden_state: torch.Tensor,    # (B, 1, H) — single position
        input_id: torch.Tensor,        # (B, 1)
        position: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample one draft token — used at inference time."""
        h   = self.hidden_proj(hidden_state)
        tok = self.tok_emb(input_id)
        pos = self.pos_emb(torch.tensor([[position]], device=input_id.device))

        fusion_input = torch.cat([h, tok], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(fusion_input))
        x    = gate * h + (1.0 - gate) * tok + pos

        for layer in self.layers:
            x = layer(x)

        x      = self.norm(x)
        logits = self.head(x[:, -1, :])   # (B, V)

        # Mask non-image tokens
        mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
        mask[self.cfg.image_token_start:self.cfg.image_token_end] = False
        logits[:, mask] = float('-inf')

        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        return torch.multinomial(probs, num_samples=1)   # (B, 1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImageTextDataset(Dataset):
    """Simple folder-based dataset.

    Expects:
        data_dir/
            image_000.jpg   (or .png, .webp)
            image_000.txt   (text prompt, same stem)
            image_001.jpg
            image_001.txt
            ...

    Returns tokenized prompt + image token sequence ready for drafter training.
    """
    def __init__(
        self,
        data_dir: str,
        processor: VLChatProcessor,
        max_image_tokens: int = 576,
    ):
        self.processor         = processor
        self.max_image_tokens  = max_image_tokens

        image_exts = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        self.image_paths = []
        for ext in image_exts:
            self.image_paths.extend(sorted(glob.glob(os.path.join(data_dir, ext))))

        # Filter to only those with a matching .txt file
        self.pairs = []
        for img_path in self.image_paths:
            txt_path = Path(img_path).with_suffix(".txt")
            if txt_path.exists():
                self.pairs.append((img_path, str(txt_path)))

        print(f"Dataset: {len(self.pairs)} image-text pairs found in {data_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        img_path, txt_path = self.pairs[idx]

        with open(txt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

        image = Image.open(img_path).convert("RGB")

        # Build conversation in Janus-Pro format
        conversation = [
            {"role": "<|User|>", "content": prompt},
            {"role": "<|Assistant|>", "content": ""},
        ]

        sft_format = self.processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.processor.sft_format,
            system_prompt="",
        )
        prompt_with_tag = sft_format + self.processor.image_start_tag

        # Tokenize prompt — we only need the prompt ids for context
        prompt_ids = self.processor.tokenizer(
            prompt_with_tag,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.squeeze(0)

        # Tokenize image — get VQ token ids for the image
        # The processor encodes the image and returns image_token ids
        inputs = self.processor(
            conversations=conversation,
            images=[image],
            force_batchify=True,
        )

        # Extract image token ids from the full sequence
        # They appear after image_start_tag in the input_ids
        full_ids      = inputs.input_ids.squeeze(0)
        img_start_tok = self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.image_start_tag
        )
        img_end_tok   = self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.image_end_tag
        )

        # Find image token span
        start_pos = (full_ids == img_start_tok).nonzero(as_tuple=True)[0]
        end_pos   = (full_ids == img_end_tok).nonzero(as_tuple=True)[0]

        if len(start_pos) == 0 or len(end_pos) == 0:
            # Fallback — return empty if image tokens not found
            return None

        img_token_ids = full_ids[start_pos[0] + 1 : end_pos[0]]
        img_token_ids = img_token_ids[:self.max_image_tokens]

        return {
            "prompt_ids":    prompt_ids,
            "img_token_ids": img_token_ids,
            "pixel_values":  inputs.pixel_values if hasattr(inputs, "pixel_values") else None,
            "inputs":        inputs,
        }


def collate_fn(batch):
    """Filter None items and pad sequences."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return batch   # return as list — handled in training loop per-sample


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_hidden_states(
    target: MultiModalityCausalLM,
    processor: VLChatProcessor,
    inputs: dict,
    img_token_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run target model forward pass and extract last-layer hidden states
    at image token positions.

    Returns:
        hidden_states: (T, H) — one hidden state per image token position
    """
    target.eval()

    # Prepare input embeddings via target's own method
    inputs_embeds = target.prepare_inputs_embeds(
        input_ids      = inputs.input_ids.to(device),
        pixel_values   = inputs.pixel_values.to(device) if inputs.pixel_values is not None else None,
        images_seq_mask= inputs.images_seq_mask.to(device) if hasattr(inputs, 'images_seq_mask') else None,
        images_emb_mask= inputs.images_emb_mask.to(device) if hasattr(inputs, 'images_emb_mask') else None,
    )

    # Full forward pass with output_hidden_states=True
    outputs = target.language_model(
        inputs_embeds=inputs_embeds,
        output_hidden_states=True,
        use_cache=False,
    )

    # Last layer hidden states: (1, full_seq_len, H)
    last_hidden = outputs.hidden_states[-1].squeeze(0)   # (full_seq_len, H)

    # Find image token positions in the full sequence
    full_ids  = inputs.input_ids.squeeze(0).to(device)
    img_start = processor.tokenizer.convert_tokens_to_ids(processor.image_start_tag)
    img_end   = processor.tokenizer.convert_tokens_to_ids(processor.image_end_tag)

    start_pos = (full_ids == img_start).nonzero(as_tuple=True)[0]
    end_pos   = (full_ids == img_end).nonzero(as_tuple=True)[0]

    if len(start_pos) == 0 or len(end_pos) == 0:
        return None

    # Hidden states at image positions — shift by 1 (predict next from current)
    img_hs = last_hidden[start_pos[0] + 1 : end_pos[0]]   # (T, H)
    img_hs = img_hs[:img_token_ids.shape[0]]

    return img_hs


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Load target model (frozen)
    # ------------------------------------------------------------------
    print(f"Loading target model: {args.target_model}")
    config = AutoConfig.from_pretrained(args.target_model, trust_remote_code=True)
    language_config = config.language_config
    language_config._attn_implementation = "eager"

    target: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        language_config=language_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()

    # Freeze all target parameters
    for p in target.parameters():
        p.requires_grad = False

    processor = VLChatProcessor.from_pretrained(args.target_model)

    # Get target hidden dimension from language model config
    target_hidden_size = target.language_model.config.hidden_size
    print(f"Target hidden size: {target_hidden_size}")

    # ------------------------------------------------------------------
    # Build drafter
    # ------------------------------------------------------------------
    drafter_cfg = DrafterConfig(
        hidden_size       = target_hidden_size,
        drafter_dim       = args.drafter_dim,
        num_heads         = args.drafter_heads,
        num_layers        = args.drafter_layers,
        vocab_size        = target.language_model.config.vocab_size,
        image_token_start = args.image_token_start,
        image_token_end   = args.image_token_end,
        max_seq_len       = args.max_image_tokens,
    )

    drafter = JanusDrafter(drafter_cfg).to(device)
    n_params = sum(p.numel() for p in drafter.parameters()) / 1e6
    print(f"Drafter parameters: {n_params:.1f}M")

    # ------------------------------------------------------------------
    # Dataset and optimizer
    # ------------------------------------------------------------------
    dataset    = ImageTextDataset(args.data_dir, processor, args.max_image_tokens)
    dataloader = DataLoader(
        dataset,
        batch_size  = 1,   # process one image at a time due to variable seq length
        shuffle     = True,
        num_workers = 4,
        collate_fn  = collate_fn,
        pin_memory  = True,
    )

    optimizer = AdamW(
        drafter.parameters(),
        lr           = args.lr,
        weight_decay = 0.01,
        betas        = (0.9, 0.95),
    )

    total_steps = args.epochs * len(dataloader)
    scheduler   = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0
    best_loss   = float("inf")

    for epoch in range(args.epochs):
        drafter.train()
        epoch_loss   = 0.0
        epoch_steps  = 0

        for batch_idx, batch in enumerate(dataloader):
            if batch is None:
                continue

            # Process each sample individually (variable sequence lengths)
            batch_loss = torch.tensor(0.0, device=device, requires_grad=False)
            valid_samples = 0

            for sample in batch:
                img_token_ids = sample["img_token_ids"].to(device)   # (T,)
                inputs        = sample["inputs"]

                if img_token_ids.shape[0] < 2:
                    continue   # need at least 2 tokens for a prediction pair

                # Extract target hidden states (no grad — target is frozen)
                with torch.no_grad():
                    hidden_states = extract_hidden_states(
                        target, processor, inputs, img_token_ids, device
                    )

                if hidden_states is None or hidden_states.shape[0] < 2:
                    continue

                hidden_states = hidden_states.to(torch.float32)   # drafter uses fp32
                T = min(hidden_states.shape[0], img_token_ids.shape[0])
                hidden_states = hidden_states[:T]
                img_token_ids = img_token_ids[:T]

                # Add batch dimension
                hs  = hidden_states.unsqueeze(0)      # (1, T, H)
                ids = img_token_ids.unsqueeze(0)      # (1, T)

                # Forward through drafter
                out  = drafter(hidden_states=hs, input_ids=ids, labels=ids)
                loss = out["loss"]

                if loss.isnan() or loss.isinf():
                    continue

                loss = loss / args.grad_accum
                loss.backward()
                batch_loss   = batch_loss + loss.detach()
                valid_samples += 1

            if valid_samples == 0:
                continue

            # Gradient clipping and optimizer step
            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(drafter.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step_loss    = batch_loss.item() * args.grad_accum
            epoch_loss  += step_loss
            epoch_steps += 1
            global_step += 1

            if global_step % args.log_every == 0:
                avg_loss = epoch_loss / max(epoch_steps, 1)
                lr_now   = scheduler.get_last_lr()[0]
                print(
                    f"epoch={epoch+1}/{args.epochs}  "
                    f"step={global_step}  "
                    f"loss={step_loss:.4f}  "
                    f"avg_loss={avg_loss:.4f}  "
                    f"lr={lr_now:.2e}"
                )

            if global_step % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, f"drafter_step{global_step}.pt")
                torch.save({
                    "step":         global_step,
                    "drafter_cfg":  drafter_cfg.__dict__,
                    "state_dict":   drafter.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

        # End of epoch
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"=== Epoch {epoch+1} complete | avg_loss={avg_epoch_loss:.4f} ===")

        if avg_epoch_loss < best_loss:
            best_loss  = avg_epoch_loss
            best_path  = os.path.join(args.output_dir, "drafter_best.pt")
            torch.save({
                "step":        global_step,
                "drafter_cfg": drafter_cfg.__dict__,
                "state_dict":  drafter.state_dict(),
            }, best_path)
            print(f"New best model saved: {best_path} (loss={best_loss:.4f})")

    print("Training complete.")


# ---------------------------------------------------------------------------
# Inference helper — how to use the trained drafter at decode time
# ---------------------------------------------------------------------------

class JanusDrafterInference:
    """Wraps the trained drafter for use in speculative decoding.

    Usage in your evaluate_posterior:
        drafter_inf = JanusDrafterInference(drafter, cfg)
        draft_token, draft_prob = drafter_inf.draft_one(
            hidden_state=h_t,    # (1, 1, 4096) from target last layer
            input_id=token_t,    # (1, 1) current token
            position=t,
        )
    """
    def __init__(self, drafter: JanusDrafter, cfg: DrafterConfig):
        self.drafter = drafter.eval()
        self.cfg     = cfg

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: torch.device) -> "JanusDrafterInference":
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg  = DrafterConfig(**ckpt["drafter_cfg"])
        model = JanusDrafter(cfg).to(device)
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, cfg)

    @torch.no_grad()
    def draft_k_tokens(
        self,
        initial_hidden: torch.Tensor,   # (1, 1, H) — hidden state at last accepted token
        initial_token: torch.Tensor,    # (1, 1) — last accepted token id
        k: int,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively draft k tokens.

        Returns:
            draft_tokens: (1, k) int64
            draft_probs:  (1, k, V) float — full distribution at each position
        """
        tokens = []
        probs  = []

        h   = initial_hidden
        tid = initial_token

        # We feed the growing draft sequence through the drafter
        # In EAGLE style, we use the target's hidden state only at the start
        # and then the drafter's own predictions for subsequent positions
        all_hs  = h
        all_ids = tid

        for step in range(k):
            out    = self.drafter(hidden_states=all_hs, input_ids=all_ids)
            logits = out["logits"][:, -1, :]   # (1, V) — last position

            # Mask non-image tokens
            mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
            mask[self.cfg.image_token_start:self.cfg.image_token_end] = False
            logits[:, mask] = float('-inf')

            prob  = F.softmax(logits / max(temperature, 1e-6), dim=-1)
            token = torch.multinomial(prob, num_samples=1)   # (1, 1)

            tokens.append(token)
            probs.append(prob.unsqueeze(1))   # (1, 1, V)

            # For next step: use a zero hidden state as placeholder
            # (the drafter learned to rely on tok_emb for continuation steps)
            zero_h  = torch.zeros(1, 1, self.cfg.hidden_size, device=h.device, dtype=h.dtype)
            all_hs  = torch.cat([all_hs, zero_h], dim=1)
            all_ids = torch.cat([all_ids, token], dim=1)

        draft_tokens = torch.cat(tokens, dim=1)           # (1, k)
        draft_probs  = torch.cat(probs, dim=1)            # (1, k, V)
        return draft_tokens, draft_probs


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train EAGLE drafter for Janus-Pro")

    # Paths
    p.add_argument("--target_model", type=str, default="deepseek-ai/Janus-Pro-7B")
    p.add_argument("--data_dir",     type=str, required=True,
                   help="Directory with image + .txt pairs")
    p.add_argument("--output_dir",   type=str, default="./drafter_ckpt")
    p.add_argument("--resume",       type=str, default=None,
                   help="Path to checkpoint to resume from")

    # Drafter architecture
    p.add_argument("--drafter_dim",    type=int, default=1024)
    p.add_argument("--drafter_heads",  type=int, default=16)
    p.add_argument("--drafter_layers", type=int, default=1)

    # Janus-Pro image token range
    p.add_argument("--image_token_start", type=int, default=100015)
    p.add_argument("--image_token_end",   type=int, default=116399)
    p.add_argument("--max_image_tokens",  type=int, default=576)

    # Training
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--batch_size", type=int,   default=4)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--grad_accum", type=int,   default=4)
    p.add_argument("--log_every",  type=int,   default=50)
    p.add_argument("--save_every", type=int,   default=500)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)