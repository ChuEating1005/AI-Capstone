import os
import shutil
import random
from pathlib import Path
import numpy as np

def main():
    # Define paths
    SOURCE_DIR = Path('images')
    OUTPUT_DIR = Path('data')

    # Define split ratios
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0
    TEST_RATIO = 0.2

    # Create output directories
    train_dir = OUTPUT_DIR / 'train'
    test_dir = OUTPUT_DIR / 'test'

    # Create directories if they don't exist
    for dir_path in [train_dir, test_dir]:
        for category in os.listdir(SOURCE_DIR):
            if category.startswith('.'):  # Skip hidden files like .DS_Store
                continue
            os.makedirs(dir_path / category, exist_ok=True)

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Process each category
    for category in os.listdir(SOURCE_DIR):
        if category.startswith('.'):  # Skip hidden files like .DS_Store
            continue
        
        category_dir = SOURCE_DIR / category
        if not os.path.isdir(category_dir):
            continue
        
        # Get all image files
        image_files = [f for f in os.listdir(category_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Shuffle the files
        random.shuffle(image_files)
        
        # Calculate split indices
        n_images = len(image_files)
        n_train = 40
        
        # Split the files
        train_files = image_files[:n_train]
        test_files = image_files[n_train:]
        
        # Copy files to respective directories
        for files, target_dir in zip([train_files, test_files], [train_dir, test_dir]):
            for file in files:
                src_path = category_dir / file
                dst_path = target_dir / category / file
                shutil.copy2(src_path, dst_path)
        
        print(f"Category {category}: {len(train_files)} train, {len(test_files)} test")

    print("Dataset preparation completed!")

if __name__ == "__main__":
    main() 