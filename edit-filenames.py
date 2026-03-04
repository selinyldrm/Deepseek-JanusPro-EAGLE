import os
from pathlib import Path
from tqdm import tqdm

def rename_files(target_dir, dry_run=True):
    path = Path(target_dir)
    
    # Find files ending in .ckpt
    # The glob pattern matches any file ending in .ckpt
    files = list(path.glob("*.npy"))
    
    if not files:
        print(f"No .ckpt files found in {target_dir}")
        return

    print(f"{'[DRY RUN]' if dry_run else '[ACTION]'} Found {len(files)} files.")

    for fpath in tqdm(files):
        # fpath.stem returns '000000000009.jpg.npy'
        # fpath.suffix returns '.ckpt'
        
        # We split by '.' and take the very first part
        # '000000000009.jpg.npy' -> '000000000009'
        new_stem = fpath.name.split('.')[0]
        new_name = f"{new_stem}{fpath.suffix}"
        
        new_path = fpath.with_name(new_name)

        if dry_run:
            print(f"Would rename: {fpath.name} -> {new_name}")
        else:
            try:
                fpath.rename(new_path)
            except Exception as e:
                print(f"Error renaming {fpath.name}: {e}")

if __name__ == "__main__":
    # 1. SET YOUR DIRECTORY HERE
    MY_DIR = "/work1/deming/shared/llamagen/mscoco_train_dataset/tokenized/text_features"
    
    # 2. RUN FIRST WITH dry_run=True to verify
    # 3. SET dry_run=False to actually rename the files
    rename_files(MY_DIR, dry_run=False)