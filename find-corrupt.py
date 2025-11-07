from PIL import Image
import os

def find_corrupted_images(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            with Image.open(file_path) as img:
                img.verify()  # Check if it's a valid image file
        except Exception as e:
            print(f"Corrupted file: {file_path} | Error: {e}")

# Replace with your actual folder path
find_corrupted_images("/work1/deming/shared/anole/loss-scaled/plain-drafter-mi250")
