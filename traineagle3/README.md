# Eagle 3 for Image Generation Models

This directory contains the Eagle 3 implementation adapted for image generation models in the LANTERN framework.

## 🔥 Key Eagle 3 Improvements

### **1. Multi-Layer Feature Extraction**
- **Eagle 3**: Extracts hidden states from **3 different layers** of the target model
- **Original**: Only uses 2 features (embeddings + single hidden state)
- **Implementation**: `hidden_states0`, `hidden_states1`, `hidden_states2` concatenated

### **2. Multi-Step Iterative Training**
- **Eagle 3**: 7 sequential training steps with progressive learning
- **Original**: Single-pass training
- **Benefits**: Better learning of complex dependencies

### **3. Progressive Attention Masking**
- **Eagle 3**: Attention mask updates at each training step
- **Implementation**: `attention_mask[:, :, ind0, ind1] = torch.finfo(attention_mask.dtype).min`

### **4. Weighted Loss Combination**
- **Eagle 3**: Exponentially weighted loss across steps
- **Formula**: `loss_weights = [0.8^i for i in range(7)]`
- **Benefits**: Earlier steps get higher weight for better guidance

## 📁 Files Overview

- `cnets_lumina_mgpt.py`: Eagle 3 CNets architecture with 3-feature input
- `main.py`: Training script with multi-step iterative training
- `config.json`: Model configuration with Eagle 3 parameters
- `training_config.yaml`: Training hyperparameters and settings
- `README.md`: This documentation

## 🚀 How to Use

### **1. Basic Training**
```bash
cd traineagle3
python main.py \
    --model lumina_mgpt \
    --base_path ckpts/lumina_mgpt/Lumina-mGPT-7B-768 \
    --data_dir /path/to/your/training/data \
    --save_dir ckpts/lumina_mgpt/trained_drafters_eagle3 \
    --num_epochs 20 \
    --bs 4 \
    --lr 1e-4
```

### **2. With CFG Training**
```bash
python main.py \
    --model lumina_mgpt \
    --coupled \
    --cfg_loss \
    --cfg_scale 3.0 \
    --base_path ckpts/lumina_mgpt/Lumina-mGPT-7B-768 \
    --data_dir /path/to/your/training/data
```

### **3. With WandB Logging**
```bash
python main.py \
    --model lumina_mgpt \
    --wandb \
    --base_path ckpts/lumina_mgpt/Lumina-mGPT-7B-768 \
    --data_dir /path/to/your/training/data
```

## 🔧 Key Parameters

### **Eagle 3 Specific:**
- `--length 7`: Number of multi-step training iterations
- `--eagle3_weight_decay 0.8`: Weight decay factor for loss combination

### **Model Configuration:**
- `--model`: Model type (lumina_mgpt, anole, llamagen)
- `--base_path`: Path to pretrained model
- `--config_path`: Path to Eagle 3 config file

### **Training Options:**
- `--coupled`: Use CFG training data
- `--cfg_loss`: Apply CFG loss
- `--embed_upscale`: Scale embedding features

## 📊 Architecture Comparison

| Feature | Original LANTERN | Eagle 3 |
|---------|------------------|---------|
| Input Features | 2 (embedding + hidden) | 3 (3 layer hiddens) |
| Training Steps | 1 | 7 |
| Loss Weighting | Single | Exponential decay |
| Attention Mask | Static | Progressive |
| Target Model Usage | Feature extraction | Multi-layer extraction |

## 🔬 Technical Details

### **Feature Extraction Process:**
```python
# Eagle 3: Extract from 3 layers
outs = self.target_model(input_ids=input_ids, output_hidden_states=True)
hidden_states0 = outs.hidden_states[0]  # Early features
hidden_states1 = outs.hidden_states[1]  # Middle features  
hidden_states2 = outs.hidden_states[2]  # Late features
hidden_states = torch.cat((hidden_states0, hidden_states1, hidden_states2), dim=-1)
```

### **Multi-Step Training Loop:**
```python
for idx in range(self.length):  # 7 steps
    # Forward pass with current features
    layer_outputs = self.midlayer(input_emb, hidden_states, ...)
    
    # Compute loss for this step
    loss = compute_step_loss(layer_outputs, target)
    plosses.append(loss)
    
    # Progressive attention masking for next step
    if not last_step:
        update_attention_mask(attention_mask, idx)
```

### **Weighted Loss Combination:**
```python
loss_weights = [0.8 ** i for i in range(7)]  # [1.0, 0.8, 0.64, ...]
final_loss = sum([loss_weights[i] * plosses[i] for i in range(len(plosses))])
```

## 🎯 Expected Improvements

Based on Eagle 3 results, you should expect:

1. **Better Draft Quality**: Multi-layer features provide richer representations
2. **Improved Training Stability**: Progressive learning with weighted losses
3. **Higher Acceptance Rates**: Better prediction accuracy from iterative refinement

## 🔍 Monitoring Training

The training script logs detailed metrics:

- `train/ploss_step_i`: Loss at each training step
- `train/acc_step_i`: Accuracy at each training step  
- `train/weighted_loss`: Final combined loss
- `train/acc`: Overall accuracy

## 🚨 Important Notes

1. **Memory Usage**: Eagle 3 uses more GPU memory due to:
   - Target model loading for feature extraction
   - Multi-step computation
   - 3x larger input features

2. **Training Time**: Expect ~7x longer training time due to multi-step process

3. **Compatibility**: Designed for Lumina-mGPT, but adaptable to other models

4. **Data Format**: Uses same data format as original LANTERN training

## 🐛 Troubleshooting

### Memory Issues:
- Reduce batch size (`--bs`)
- Use gradient accumulation (`--gradient_accumulation_steps`)
- Enable gradient checkpointing (default: enabled)

### Training Instability:
- Reduce learning rate (`--lr`)
- Adjust weight decay factor (`--eagle3_weight_decay`)
- Use mixed precision training (default: bf16)

### Performance Issues:
- Monitor individual step losses
- Check if early steps are learning properly
- Adjust the number of training steps (`--length`)

## 📚 References

- Original Eagle Paper: [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)
- Eagle 3 Implementation: Based on the Eagle 3 training strategy
- LANTERN Paper: [LANTERN: Accelerating Visual Autoregressive Models via Relaxed Speculative Decoding](https://arxiv.org/abs/2410.03355)

