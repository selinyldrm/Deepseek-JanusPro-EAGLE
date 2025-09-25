import os
import shutil

# Output directory to collect renamed images
DEST_DIR = "./all-imgs"
os.makedirs(DEST_DIR, exist_ok=True)

# Supported image file extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}

# Go through each item in the current directory
for folder in os.listdir('.'):
    folder_path = os.path.join('.', folder)
    if os.path.isdir(folder_path):
        # List files with image extensions
        image_files = [f for f in os.listdir(folder_path)
                       if os.path.isfile(os.path.join(folder_path, f)) and
                          os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]

        if len(image_files) != 1:
            print(f"⚠️ Skipping '{folder}' — expected 1 image, found {len(image_files)}")
            continue

        image_file = image_files[0]
        ext = os.path.splitext(image_file)[1].lower()
        src_path = os.path.join(folder_path, image_file)
        dest_path = os.path.join(DEST_DIR, folder + ext)

        shutil.copy2(src_path, dest_path)
        print(f"✅ Copied: {src_path} → {dest_path}")
