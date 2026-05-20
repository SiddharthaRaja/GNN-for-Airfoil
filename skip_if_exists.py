"""
skip preprocessing if data already exists.
useful for resuming work without re-downloading.
"""
import os
import sys
import torch

def check_and_skip():
    """check if preprocessed data exists and skip if it does"""
    data_dir = './data'
    required_files = ['train_dataset.pt', 'val_dataset.pt', 'test_dataset.pt', 'norm_stats.pt']
    
    print("Checking for existing preprocessed data...")
    
    all_exist = all(os.path.exists(os.path.join(data_dir, f)) for f in required_files)
    
    if all_exist:
        print("all preprocessed data files found!")
        print("  Location: ./data/")
        print("\nYou can skip preprocessing and go directly to training:")
        print("  python train_gcn.py")
        print("\nOr if you want to regenerate, delete ./data/ folder first.")
        return True
    else:
        missing = [f for f in required_files if not os.path.exists(os.path.join(data_dir, f))]
        print(f"missing files: {', '.join(missing)}")
        print("  Run preprocessing first:")
        print("  python data_preprocess.py")
        print("  OR for quick test: python quick_preprocess.py")
        return False

if __name__ == '__main__':
    if check_and_skip():
        sys.exit(0)
    else:
        sys.exit(1)

