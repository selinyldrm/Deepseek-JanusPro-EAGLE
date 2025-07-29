
# LANTERN

This repository is an official PyTorch implementation of the paper [LANTERN: Accelerating Visual Autoregressive Models via Relaxed Speculative Decoding](https://arxiv.org/abs/2410.03355) (ICLR 2025) and [LANTERN++: Enhanced Relaxed Speculative Decoding with Static Tree Drafting for Visual Auto-regressive Models](https://arxiv.org/abs/2502.06352) (ICLRW - SCOPE(Oral) 2025), which supports various functionalities related to LANTERN, including model inference, drafter model training, drafter model training data generation and image decoding for image generation.

---

## 📰 News

- **[2025-03-05] 🎉🎉🎉 LANTERN is released! 🎉🎉🎉**

---

## 📚 Table of Contents
1. [Directory Structure](#directory-structure)
2. [Installation](#installation)
3. [Key Features](#key-features)
4. [Usage](#usage)
5. [License](#license)
6. [Acknowledgement](#acknowledgement)
7. [Citation](#citation)
---

## 🗂️ Directory Structure

The main directory structure of the project is as follows:

```
.
├── models/                                     # Model and related modules
│   ├── base_models/                            # Base model modules
│   │   ├── lumina_mgpt
│   │   │   ├── modeling_lumina_mgpt.py
│   │   │   └── other files...
│   │   └── other models...     
│   ├── kv_variants/                            # Key-Value variant models
│   │   ├── modeling_lumina_mgpt_kv.py
|   |   └── modeling_anole_kv.py
│   │   └── other models...
│   ├── drafters/                               # Drafter model modules
│   │   ├── kv_cache.py
│   │   ├── choices
│   │   ├── cnets_lumina_mgpt.py
|   |   ├── cnets_anole.py
│   │   ├── cnets_{other_models}.py ...   
│   │   └── utils.py
│   ├── configs/                                # Configuration modules
│   │   ├── configs.py
│   │   ├── configuration_lumina_mgpt.py
|   |   ├── configuration_anole.py
│   │   └── configuration_{other_models}.py...
│   ├── ea_model_lumina_mgpt.py                 # EAGLE models
|   ├── ea_model_anole.py
│   └── ea_model_{other_models}.py...
├── data/
│   ├── configs/
│   │   ├── lumina_mgpt_config.json             # Configuration for model init
|   |   ├── anole_config.json
│   │   └── configs for other models...
│   ├── prompts/                                # Prompts for image generation
│   ├── self_distilled_data/                    # Self-distilled data for drafter training
│   └── drafter_train_data/                     # Train data for drafter
├── ckpts/                                      # Model checkpoints folder
│   ├── lumina_mgpt/
│   │   ├── chameleon/                      
│   │   ├── Lumina-mGPT-7B-768/                 # Model and tokenizer files
│   │   ├── trained_drafters/                   # Trained drafter models
|   |   |   └──...state_20/
|   |   |      ├── config.json                  # config.json for drafter model
|   |   |      └── other files...
│   │   └── vq_distances/                       # Pre-computed VQ distances for LANTERN
│   └── other models...
├── entrypoints/                                # Execution entry points
│   ├── train_drafter/
│   │   ├── data_utils.py
│   │   └── main.py
│   ├── generate_codebook.py
│   ├── generate_images.py
│   ├── generate_train_data.py
│   └── other files...
├── third_party/                                # Third-party libraries
│   └── vllm
├── main.py                                     # Main execution script
├── requirements.txt                            # Project dependencies
├── environment.yaml
├── .gitignore
└── README.md
```

Here is a brief description for each directory.
1. **`models/`** - Contains model implementations and related modules.
   - **`base_models/`** - Base model implementations (e.g., Lumina-mGPT, LlamaGen, Anole).
   - **`kv_variants/`** - Modified base models with Key-Value cache adaptations for enhanced compatibility with EAGLE’s architecture.
   - **`drafters/`** - Modules and auxiliary code for drafter models.
   - **`configs/`** - Configuration modules for each model (e.g., `ChameleonConfig` for Lumina-mGPT).

2. **`data/`** - Stores configuration files, text prompts, self-distilled data, and drafter training data.

3. **`ckpts/`** - Checkpoints for all models, including trained drafters and VQ distances for relaxed speculative decoding.

4. **`entrypoints/`** - Primary scripts for tasks such as image generation, codebook generation, and drafter training.

5. **`third_party/`** - Custom external libraries, including modifications for specific functionality.

---

## ⚙️ Installation

1. **Install Required Packages**
    **Requirements**
    - Python >= 3.10
    - PyTorch >= 2.4.0
    
    Install the dependencies listed in `requirements.txt`.
    ```bash
    git clone https://github.com/jadohu/LANTERN
    cd LANTERN
    pip install -r requirements.txt
    ```

2. **Additional Setup**
    1. **Lumina-mGPT**
        For [Lumina-mGPT](https://github.com/Alpha-VLLM/Lumina-mGPT), we need to install `flash_attention` and `xllmx` packages.
        ```bash
        pip install flash-attn --no-build-isolation
        cd models/base_models/lumina_mgpt
        pip install -e .
        ```
        1. **(Optional) vLLM**
            Install and set up [`vLLM`](https://github.com/vllm-project/vllm) with the required modifications. Note that we use `vLLM==0.6.3` and build from source. The required modifications are specifed in `third_party/vllm`. The installation procedure is as follows.
            ```bash
            pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/fd47e57f4b0d5f7920903490bce13bc9e49d8dba/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
            git clone https://github.com/vllm-project/vllm
            cd vllm
            git checkout tags/v0.6.3

            cd ..
            mv -rf third_party/vllm/* vllm/
            cd vllm
            python python_only_dev.py
            ```

3. **Checkpoints**
    All model weights and other required data should be stored in `ckpts/`.
    1. **Lumina-mGPT**
        For Lumina-mGPT, since currently the Chameleon implementation in transformers does not contain the VQ-VAE decoder, please manually download the original VQ-VAE weights [provided by Meta](https://github.com/facebookresearch/chameleon) and put them to the following directory:
        ```
        ckpts
        └── lumina_mgpt
            └── chameleon
                └── tokenizer
                    ├── text_tokenizer.json
                    ├── vqgan.yaml
                    └── vqgan.ckpt
        ```

        Also download the original model [`Lumina-mGPT-7B-768`](https://huggingface.co/Alpha-VLLM/Lumina-mGPT-7B-768) from Huggingface 🤗 and put them to the following directory:
        ```
        ckpts
        └── lumina_mgpt
            └── Lumina-mGPT-7B-768
                ├── config.json
                ├── generation_config.json
                ├── model-00001-of-00002.safetensors
                └── other files...
        ```
    2. **LlamaGen**
        For LlamaGen T2I model, download [`LlamaGen-T2I`](https://huggingface.co/jadohu/LlamaGen-T2I) and/or [`LlamaGen-T2I-2`](https://huggingface.co/jadohu/LlamaGen-T2I-2), which is a huggingface style converted model from [`LlamaGen`](https://github.com/FoundationVision/LlamaGen). 
        
        In addition, you should download [`VQ-VAE`](https://huggingface.co/peizesun/llamagen_t2i/resolve/main/vq_ds16_t2i.pt) and [`flan-t5-xl`](https://huggingface.co/google/flan-t5-xl).

        ```
        ckpts
        └── llamagen
            ├── LlamaGen-T2I
            │   ├── config.json
            │   ├── generation_config.json
            │   ├── model.safetensors
            │   └── other files...
            ├── LlamaGen-T2I-2
            │   ├── config.json
            │   ├── generation_config.json
            │   ├── model.safetensors
            │   └── other files...
            ├── vq_ds16_t2i.pt
            └── t5
                └── flan-t5-xl
                    ├── config.json
                    ├── generation_config.json
                    ├── model-00001-of-00002.safetensors
                    └── other files...
        ```

        **(Optional) Trained drafter**
        To use trained drafter, you need to download [`llamagen_drafter`](https://huggingface.co/jadohu/llamagen_drafter) and/or [`llamagen2_drafter`](https://huggingface.co/jadohu/llamagen2_drafter) and save it under trained_drafters directory.
        ```
        ckpts
        └── llamagen
            └── trained_drafters
                ├── llamagen_drafter
                |   ├── config.json
                |   ├── generation_config.json
                |   ├── pytorch_model.bin
                |   └── other files...
                └── llamagen2_drafter
                    ├── config.json
                    ├── generation_config.json
                    ├── pytorch_model.bin
                    └── other files...
        ```
    3. **Anole**
        For Anole, download [`Anole-7b-v0.1-hf`](https://huggingface.co/leloy/Anole-7b-v0.1-hf), which is a huggingface style converted model from [`Anole`](https://huggingface.co/GAIR/Anole-7b-v0.1). 
        
        In addition, you should download the original VQ-VAE weights [provided by Meta](https://github.com/facebookresearch/chameleon) and put them to the following directory:

        ```
        ckpts
        └── anole
            ├── Anole-7b-v0.1-hf
            |   ├── config.json
            |   ├── generation_config.json
            |   ├── model-00001-of-00003.safetensors
            |   └── other files...
            └── chameleon
                └── tokenizer
                    ├── text_tokenizer.json
                    ├── vqgan.yaml
                    └── vqgan.ckpt
        ```

        **(Optional) Trained drafter**
        To use trained drafter, you need to download [`anole_drafter`](https://huggingface.co/jadohu/anole_drafter) and save it under trained_drafters directory.
        ```
        ckpts
        └── anole
            └── trained_drafters
                └── anole_drafter
                    ├── config.json
                    ├── generation_config.json
                    ├── pytorch_model.bin
                    └── other files...
        ```

---

## ✨ Usage

All the functionalities can be done by either running `main.py` or directly running `entrypoints/{function}.py`.
Currently, "llamagen" (LlamaGen-Stage I), "llamagen2" (LlamaGen-Stage II), "anole", and "lumina_mgpt" are supported as --model.

🚧 **Lumina-mGPT is still under construction, so some functions may not work properly yet. You can follow the procedures here, but you may encounter a few exceptions.**

1. **Generate Images**
    ```bash
    python main.py generate_images --model <model_name> --model_type <model_type; e.g., base, vllm, eagle> --model_path <model_path> --drafter_path <drafter_path> --output_dir <output_dir> ...
    ```
    or
     ```bash
     python -m entrypoints.generate_images --model <model_name> --model_type <model_type; e.g., base, vllm, eagle> --model_path <model_path> --drafter_path <drafter_path> --output_dir <output_dir> ...
     ```

    💡**How to use LANTERN and LANTERN++ for image generation**
    - For **LANTERN**, set `--model_type eagle`, turn on `--lantern` option and set `--lantern_k` and `--lantern_delta` options.
    - For **LANTERN++**, use `--static_tree` option and use `--lantern_delta` to set $\lambda$ value. 

2. **Generate Training Data for Drafter**
    ```bash
    python main.py generate_train_data --model <model_name> --data_path <path_to_image_tokens> --output_dir <output_dir> --num_samples <num_samples>
    ```
    or
     ```bash
     python -m entrypoints.generate_train_data --model <model_name> --data_path <path_to_image_tokens> --output_dir <output_dir> --num_samples <num_samples>
     ```

    For **LlamaGen** and **Anole**, you have to extract code and T5 embedding(only for LlamaGen) for training data. 
    - Locate image and caption files in given format and execute following command before run **generate_train_data**:

    **Data Format:**
    - image_folder
        - {file_1}.jpg
        - {file_1}.txt
        - {file_2}.jpg
        - {file_2}.txt
        ... 
    ```bash
    python main.py extract_code --model <model_type> --data_path <path_to_image_and_caption> --output_dir <output_dir> --num_samples <num_samples>
    ```
    or 
    ```bash
    python -m entrypoints.extract_code --model <model_type> --data_path <path_to_image_and_caption> --output_dir <output_dir> --num_samples <num_samples>
    ```
3. **Train Drafter Model**
   ```bash
    python main.py train_drafter --model <model_type> --base_path <base_model_path> --config_path <path_to_config.json> --data_dir <data_dir> --save_dir <save_dir> --lr <lr> --bs <bs> --gradient_accumlation_steps <gradient_accumulation_steps> ...
    ```
    or
     ```bash
     python -m entrypoints.train_drafter.main --model <model_type> --base_path <base_model_path> --config_path <path_to_config.json> --data_dir <data_dir> --save_dir <save_dir> --lr <lr> --bs <bs> --gradient_accumlation_steps <gradient_accumulation_steps> ...
     ```

    For multi GPU training with accelerate, you can use
    ```bash
     accelerate launch -m entrypoints.train_drafter.main --model <model_type> --base_path <base_model_path> --config_path <path_to_config.json> --data_dir <data_dir> --save_dir <save_dir> --lr <lr> --bs <bs> --gradient_accumlation_steps <gradient_accumulation_steps> ...
     ```

4. **Generate VQ Distances**
     ```bash
     python main.py generate_codebook --model <model_name> --save_path <save_path>
     ```
     or
     ```bash
     python -m entrypoints.generate_codebook --model <model_name> --save_path <save_path>
     ```


5. **Evaluate Generated Images**
    We support FID, CLIP score, Precision/Recall and HPSv2 for image evaluation.
    ```bash
    python main.py eval_fid_clip --fake_dir <path_to_generated_image> --ref_dir <path_to_reference_image> --caption_path <path_to_prompt> --how_many <number_of_images_for_evaluation> ...
    ```
    ```bash
    python main.py eval_prec_recall --fake_dir <path_to_generated_image> --ref_dir <path_to_reference_image> ...
    ```
    ```bash
    python main.py eval_hpsv2 --image_path <path_to_generated_image> --prompt_path <path_to_prompt>
    ```
    or 
    ```bash
    python -m entrypoints.eval_fid_clip --fake_dir <path_to_generated_image> --ref_dir <path_to_reference_image> --caption_path <path_to_prompt> --how_many <number_of_images_for_evaluation> ...
    ```
    ```bash
    python -m entrypoints.eval_prec_recall --fake_dir <path_to_generated_image> --ref_dir <path_to_reference_image> ...
    ```
    ```bash
    python -m entrypoints.eval_hpsv2 --image_path <path_to_generated_image> --prompt_path <path_to_prompt>
    ```
---

## ⚠️ CAUTIONS
1. **`config.json` should be in the `ckpts/{model_name}/trained_models/{drafter_path}`**
    Since the `Model` in `cnets_{model_name}.py` is initialized according to the `config.json` in the `drafter_path`, you need to place `config.json` for drafter correctly. Note that the `config.json` should be same as the base model's `config.json` other than `num_hidden_layers`.

---

## ⚖️ License

This project is distributed under the Chameleon License by Meta Platforms, Inc. For more information, please see the `LICENSE` file in the repository.

---

## 🙏 Acknowledgement
This repository is built with extensive reference to [FoundationVision/LlamaGen](https://github.com/FoundationVision/LlamaGen), [Alpha-VLLM/Lumina-mGPT](https://github.com/Alpha-VLLM/Lumina-mGPT) and [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE), leveraging many of their core components and approaches.

---

## 📄 Citation

```
@article{jang2024lantern,
  title={LANTERN: Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding},
  author={Jang, Doohyuk and Park, Sihwan and Yang, June Yong and Jung, Yeonsung and Yun, Jihun and Kundu, Souvik and Kim, Sung-Yub and Yang, Eunho},
  journal={arXiv preprint arXiv:2410.03355},
  year={2024}
}
@article{park2025lanternenhancedrelaxedspeculative,
  title={LANTERN++: Enhanced Relaxed Speculative Decoding with Static Tree Drafting for Visual Auto-regressive Models}, 
  author={Sihwan Park and Doohyuk Jang and Sungyub Kim and Souvik Kundu and Eunho Yang},
  journal={arXiv preprint arXiv:2410.03355},
  year={2025}
}
```