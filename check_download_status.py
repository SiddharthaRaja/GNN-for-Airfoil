"""
check the status of airfrans dataset download.
use this to see if you can skip downloading.
"""
import os
import zipfile

def format_size(size_bytes):
    """convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def check_download_status(root='./airfrans_data'):
    """check the status of dataset download and extraction"""
    print("=" * 60)
    print("Airfrans Dataset Status Check")
    print("=" * 60)
    
    zip_path = os.path.join(root, 'Dataset.zip')
    manifest_path = os.path.join(root, 'manifest.json')
    
    # check for manifest (fully extracted)
    if os.path.exists(manifest_path):
        print("\ndataset status: READY")
        print(f"  Location: {manifest_path}")
        print("  The dataset is fully downloaded and extracted.")
        print("  You can proceed with preprocessing!\n")
        return "ready"
    
    # check for zip file
    if os.path.exists(zip_path):
        size = os.path.getsize(zip_path)
        size_str = format_size(size)
        expected_size_gb = 9.3
        
        print(f"\nzip file found: {zip_path}")
        print(f"  Size: {size_str}")
        print(f"  Expected: ~{expected_size_gb} GB")
        print(f"  Full path: {os.path.abspath(zip_path)}")
        
        # check if zip is valid
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.testzip()
            print("  Status: Valid zip file")
            print("  Note: If you manually placed this file, it will be used.")
            
            # check if extraction is needed
            if not os.path.exists(manifest_path):
                print("\n  Action needed: EXTRACTION REQUIRED")
                print("  The zip file needs to be extracted.")
                print("  This may take 5-15 minutes depending on your system.")
                print("\n  You can:")
                print("  1. Run: python data_preprocess.py (will extract automatically)")
                print("  2. Or extract manually using 7-Zip/WinRAR")
                print(f"     Extract to: {os.path.abspath(root)}")
                return "extract"
            else:
                return "ready"
                
        except (zipfile.BadZipFile, zipfile.LargeZipFile) as e:
            print(f"  Status: CORRUPTED")
            print(f"  Error: {e}")
            print("\n  Action needed: RE-DOWNLOAD OR REPLACE REQUIRED")
            print("  If you manually downloaded this, please re-download.")
            print("  Or delete the corrupted zip and let the script download it.")
            return "corrupted"
    
    # check for partial download
    partial_files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f)) and f.endswith('.tmp')]
    if partial_files:
        print(f"\npartial download detected: {len(partial_files)} temporary file(s)")
        print("  A download may be in progress.")
        print("  Wait for it to complete or restart the download.")
        return "downloading"
    
    # no files found
    print("\ndataset status: NOT FOUND")
    print("  No dataset files found.")
    print(f"  Expected location: {os.path.abspath(root)}")
    print("\n  Action needed: DOWNLOAD REQUIRED (~9.3 GB)")
    print("\n  Options:")
    print("  1. Manually download Dataset.zip and place it in:")
    print(f"     {os.path.abspath(root)}")
    print("  2. Let the script download automatically (will take time):")
    print("     - Fast connection (100 Mbps): ~15-20 minutes")
    print("     - Medium connection (50 Mbps): ~30-40 minutes")
    print("     - Slow connection (10 Mbps): ~2+ hours")
    print("  3. Use quick preprocessing (smaller dataset):")
    print("     python quick_preprocess.py")
    return "not_found"

def check_preprocessed_data():
    """check if preprocessed data already exists"""
    data_dir = './data'
    required_files = ['train_dataset.pt', 'val_dataset.pt', 'test_dataset.pt', 'norm_stats.pt']
    
    print("\n" + "=" * 60)
    print("Preprocessed Data Status")
    print("=" * 60)
    
    if not os.path.exists(data_dir):
        print("\npreprocessed data directory not found")
        return False
    
    existing_files = []
    missing_files = []
    
    for file in required_files:
        filepath = os.path.join(data_dir, file)
        if os.path.exists(filepath):
            size = format_size(os.path.getsize(filepath))
            existing_files.append(f"  {file} ({size})")
        else:
            missing_files.append(f"  {file}")
    
    if existing_files:
        print("\nfound preprocessed files:")
        for f in existing_files:
            print(f)
    
    if missing_files:
        print("\nmissing files:")
        for f in missing_files:
            print(f)
        return False
    
    if not missing_files:
        print("\nall preprocessed data files exist!")
        print("  You can skip preprocessing and go directly to training:")
        print("  python train_gcn.py")
        return True
    
    return False

if __name__ == '__main__':
    # check raw dataset status
    status = check_download_status()
    
    # check preprocessed data status
    has_preprocessed = check_preprocessed_data()
    
    # summary
    print("\n" + "=" * 60)
    print("Summary & Recommendations")
    print("=" * 60)
    
    if has_preprocessed:
        print("\nyou have preprocessed data! you can skip preprocessing.")
        print("  Run: python train_gcn.py")
    elif status == "ready":
        print("\ndataset is ready! run preprocessing:")
        print("  python data_preprocess.py")
        print("  OR for quick test: python quick_preprocess.py")
    elif status == "extract":
        print("\nzip file found but needs extraction.")
        print("  Run: python data_preprocess.py (will extract automatically)")
    elif status == "downloading":
        print("\ndownload may be in progress. wait or restart.")
    elif status == "corrupted":
        print("\nzip file is corrupted. delete it and re-download.")
    else:
        print("\ndataset not found. options:")
        print("  1. Quick test: python quick_preprocess.py (faster, minimal data)")
        print("  2. Full dataset: python data_preprocess.py (slower, complete data)")

