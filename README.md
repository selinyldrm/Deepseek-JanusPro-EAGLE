## 🦅 JanusPro-EAGLE: Faster Image Generation with DeepSeek Janus-Pro 🎨 ##

Welcome to JanusPro-EAGLE, where we’ve given the powerhouse DeepSeek Janus-Pro a pair of high-speed wings! By using EAGLE-based Speculative Decoding, we’re slashing image generation times without losing a single pixel of that multimodal magic.

Why wait for your masterpieces when you can speculate them into existence? 🚀

## 🌟 Features ##
Turbocharged Generation: Powered by the EAGLE framework for efficient speculative sampling.

Dual-Model Synergy: Combines the visual depth of Janus-Pro with a lightning-fast custom Drafter.

Seamless CFG: Built-in support for Classifier-Free Guidance to keep your prompts sharp and your colors popping.

## 🛠️ Installation & Setup ##
The Base Model: You'll need the Janus-Pro-7B weights.

## 🧠 The Drafter: ##
Install our specialized Drafter checkpoints from Hugging Face:
👉 [JanusPro-Eagle-Drafter](https://huggingface.co/seliny2/JanusPro-Eagle-Drafter)

## 🚀 Quick Start: ##
Use the following command to fire up the engine. 
```
python3 main.py generate_images \
    --model januspro \
    --model_type eagle \
    --model_path /your/path/to/Janus-Pro-7B \
    --drafter_path /your/path/to/Eagle-Janus-Pro \
    --prompt "A serene landscape of a digital forest at sunset, synthwave style"
Pro-Tip: If you're testing on the standard validation set, just pass --prompt MSCOCO2017Val to kick off the automated benchmark!
```
## 📂 Repo Structure ##
ea_model_januspro.py: The heart of the EAGLE-based speculative tree decoding.

main.py: Your entry point for generating images.

entrypoints/train_drafter/main.py: Everything you need to train your own drafter for JanusPro.

## 🤝 Contributing ##
Got a way to make it even faster? Found a bug? Open a PR or an issue!  🦅✨
