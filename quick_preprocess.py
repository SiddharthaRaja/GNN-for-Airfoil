"""
quick preprocessing script with minimal dataset for fast testing.
use this if the full dataset download is taking too long.
"""
import torch
from data_preprocess import preprocess_dataset
import os

def main():
    print("=" * 60)
    print("QUICK PREPROCESSING - Moderate Dataset for Fast Testing")
    print("=" * 60)
    print("\nThis script uses moderate samples (100/50/50) for quick testing.")
    print("Good balance between speed and results quality.\n")
    
    # check if data already exists
    if os.path.exists('./data/train_dataset.pt'):
        print("preprocessed data already exists!")
        print("  If you want to regenerate, delete ./data/ folder first.\n")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Skipping preprocessing.")
            return
    
    # use moderate samples for quick testing
    print("Using moderate samples for quick testing:")
    print("  - Train: 100 samples (instead of 500)")
    print("  - Val: 50 samples (instead of 100)")
    print("  - Test: 50 samples (instead of 100)\n")
    
    try:
        # preprocess training data
        print("Preprocessing training data...")
        train_dataset, norm_stats = preprocess_dataset(
            split='train', 
            n_samples=100,  # moderate size for better results
            save_dir='./data'
        )
        print(f"training data: {len(train_dataset)} samples\n")
        
        # preprocess validation data
        print("Preprocessing validation data...")
        val_dataset, _ = preprocess_dataset(
            split='val', 
            n_samples=50,
            save_dir='./data'
        )
        print(f"validation data: {len(val_dataset)} samples\n")
        
        # preprocess test data
        print("Preprocessing test data...")
        test_dataset, _ = preprocess_dataset(
            split='test', 
            n_samples=50,
            save_dir='./data'
        )
        print(f"test data: {len(test_dataset)} samples\n")
        
        print("=" * 60)
        print("quick preprocessing complete!")
        print("=" * 60)
        print("\nYou can now run:")
        print("  python train_gcn.py")
        print("\nNote: Results will be limited due to small dataset size.")
        print("For full results, use: python data_preprocess.py (with more samples)")
        
    except Exception as e:
        print(f"\nerror during preprocessing: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Make sure airfrans package is installed: pip install airfrans")
        print("3. Try running again - download may resume")
        raise

if __name__ == '__main__':
    main()

