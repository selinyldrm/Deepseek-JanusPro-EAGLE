import torch
import os

def find_broken_ckpts(folder):
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if not os.path.isfile(path) or not file.endswith(('.pt', '.ckpt')):
            continue

        if os.path.getsize(path) == 0:
            print(f"❌ Empty checkpoint: {path}")
            continue

        try:
            # More strict loading: do NOT pass weights_only=True
            torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"❌ Corrupted/invalid checkpoint: {path} | Error: {e}")

folder = "/work1/deming/shared/relaion-coco-training-data_eagle3"
find_broken_ckpts(folder)
