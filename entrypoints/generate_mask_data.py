########## change file naming convention ##################

# import os
# from pathlib import Path
# from tqdm import tqdm

# def strip_leading_zeros(mask_dir):
#     mask_path = Path(mask_dir)
    
#     # 1. Gather all .pt mask files
#     mask_files = list(mask_path.glob("*.pt"))
#     print(f"Found {len(mask_files)} masks to rename.")

#     success_count = 0
#     error_count = 0

#     # 2. Rename loop
#     for fpath in tqdm(mask_files):
#         try:
#             # Get the filename without extension (e.g., '000000000847')
#             old_name = fpath.stem
            
#             # Convert to int and back to str to remove zeros (e.g., '847')
#             new_name = str(int(old_name))
            
#             # Construct new full path
#             new_fpath = fpath.with_name(f"{new_name}.pt")
            
#             # Perform the rename
#             fpath.rename(new_fpath)
#             success_count += 1
            
#         except Exception as e:
#             print(f"Error renaming {fpath.name}: {e}")
#             error_count += 1

#     print(f"\n--- Renaming Complete ---")
#     print(f"Successfully renamed: {success_count}")
#     print(f"Errors:               {error_count}")

# if __name__ == "__main__":
#     # Path to your masks
#     MASK_DIR = "/work1/deming/shared/llamagen/mscoco_train_dataset/masks-labels"
    
#     strip_leading_zeros(MASK_DIR)
   
######### find missing file names that exists in path A but not in path B ####################
 
# import os
# from pathlib import Path

# def count_intersecting_files(data_dir, mask_dir):
#     # 1. Get all file stems from the data directory
#     # Stems are the filenames without extensions (e.g., 'sample_01')
#     data_stems = {f.stem for f in Path(data_dir).glob("*.jpg")}
    
#     # 2. Get all file stems from the mask directory
#     # (Update extension if your raw masks are .png or .jpg instead of .pt)
#     mask_stems = {f.stem for f in Path(mask_dir).glob("*.jpg")}

#     # # 3. Find the intersection
#     # intersection = data_stems.intersection(mask_stems)
    
#     # # 4. Report results
#     # print(f"--- Dataset Alignment Report ---")
#     # print(f"Total Data files:  {len(data_stems)}")
#     # print(f"Total Mask files:  {len(mask_stems)}")
#     # print(f"Common Files:      {len(intersection)}")
#     # print(f"Missing Masks:     {len(data_stems - mask_stems)}")
    
    
#     missing = list(data_stems - mask_stems)

#     # out_dir = f"{mask_dir}-temp"
#     out_dir = mask_dir
#     if len(missing) > 0 :
#         import torch.multiprocessing as mp
#         from infer import worker as worker_func
#         mp.set_start_method("spawn", force=True)

#         image_files = [f"{m}.jpg" for m in missing]
#         world_size = 8  # number of GPUs

#         mp.spawn(
#             worker_func,
#             args=(world_size, image_files, data_dir, out_dir),
#             nprocs=world_size,
#             join=True,
#         )
    
#     return intersection

# # --- Run the check ---
# if __name__ == "__main__":
#     DATA_PATH = "/work1/deming/shared/llamagen/mscoco_train_dataset/images"
#     MASK_PATH = "/work1/deming/shared/llamagen/mscoco_train_dataset/masks"
    
#     intersected_ids = count_intersecting_files(DATA_PATH, MASK_PATH)


import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial

class LlamaGenIndividualProcessor:
    def __init__(self, image_size=512, downsample_ratio=16, text_prefix_len=120):
        self.grid_size = image_size // downsample_ratio
        self.text_prefix_len = text_prefix_len
        
    def _process_single_file(self, fpath, output_path):
        """Helper function for worker processes to handle one file."""
        try:
            # Load and Binarize
            mask_img = Image.open(fpath).convert('L')
            mask_np = np.array(mask_img)
            mask_torch = torch.from_numpy(mask_np).float()
            
            # Standardize values to 0 or 1
            if mask_torch.max() > 1:
                mask_torch = (mask_torch > 127).float()
            
            # 3. Patch-Density Downsampling (Area)
            # F.interpolate expects [Batch, Channel, H, W]
            mask_4d = mask_torch.unsqueeze(0).unsqueeze(0)
            downsampled = F.interpolate(
                mask_4d, 
                size=(self.grid_size, self.grid_size), 
                mode='area'
            ).view(-1)
            
            # 5. Save with same filename but .pt extension
            save_name = fpath.stem + ".pt"
            torch.save(downsampled, output_path / save_name)
            return True
        except Exception as e:
            # Note: print statements in workers may overlap; 
            # for production, consider using the logging module.
            print(f"Skipping {fpath.name} due to error: {e}")
            return False

    def process_and_save_parallel(self, input_dir, output_dir, num_workers=None):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all common image formats
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.npy']
        mask_files = []
        for ext in extensions:
            mask_files.extend(list(input_path.glob(ext)))
            
        print(f"Parallel processing {len(mask_files)} masks into {output_dir}...")
        
        # Use partial to pass the output_path to the helper function
        worker_func = partial(self._process_single_file, output_path=output_path)

        # ProcessPoolExecutor manages the pool of worker processes
        with ProcessPoolExecutor(max_workers=16) as executor:
            # Use tqdm to track progress over the list of files
            list(tqdm(executor.map(worker_func, mask_files), total=len(mask_files)))

# --- Execution ---
if __name__ == "__main__":
    processor = LlamaGenIndividualProcessor(
        image_size=512, 
        downsample_ratio=16, 
        text_prefix_len=120
    )
    
    # Defaults to number of processors on the machine if num_workers is None
    processor.process_and_save_parallel(
        input_dir="/work1/deming/shared/llamagen/mscoco_train_dataset/masks", 
        output_dir="/work1/deming/shared/llamagen/mscoco_train_dataset/masks-labels",
        num_workers=os.cpu_count() 
    )