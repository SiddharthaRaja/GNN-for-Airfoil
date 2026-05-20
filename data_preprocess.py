import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
import numpy as np
import airfrans as af
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import zipfile
import time
import glob

def load_airfrans_data(split='train', n_samples=None, root='./airfrans_data', task='full'):
    """load airfrans dataset"""
    # check if preprocessed .pth files exist (alternative data format)
    pth_pattern = os.path.join(root, 'graph_airfrans_data_batch_*', '*.pth')
    pth_files = sorted(glob.glob(pth_pattern))
    
    if pth_files:
        print(f"found {len(pth_files)} preprocessed .pth files")
        print("loading preprocessed graph data directly...")
        
        all_data = []
        for pth_file in pth_files:
            try:
                # load with weights_only=False for PyTorch Geometric Data objects
                data = torch.load(pth_file, weights_only=False)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
            except Exception as e:
                print(f"warning: could not load {pth_file}: {e}")
                continue
        
        if len(all_data) == 0:
            raise ValueError("no valid data loaded from .pth files")
        
        print(f"loaded {len(all_data)} samples from preprocessed files")
        
        # check if these are already graph data objects (have edge_index)
        # check first few samples to be sure
        is_graph_data = False
        for sample in all_data[:min(5, len(all_data))]:
            if isinstance(sample, Data) or (hasattr(sample, 'edge_index') and hasattr(sample, 'x')):
                is_graph_data = True
                break
        
        if is_graph_data:
            print("data is already in graph format, using directly")
            if n_samples:
                return all_data[:n_samples]
            return all_data
        else:
            # if they're raw samples, convert to graphs
            print("converting samples to graph format...")
            graph_data = []
            for i, sample in enumerate(all_data):
                if i % 10 == 0:
                    print(f"processing sample {i}/{len(all_data)}")
                try:
                    graph_data.append(create_graph_data(sample))
                except Exception as e:
                    print(f"error processing sample {i}: {e}")
                    continue
            if len(graph_data) == 0:
                raise ValueError("failed to convert any samples to graph format")
            if n_samples:
                return graph_data[:n_samples]
            return graph_data
    
    # check if manifest exists, if not download/extract
    manifest_path = os.path.join(root, 'manifest.json')
    zip_path = os.path.join(root, 'Dataset.zip')
    
    # create root directory if it doesn't exist
    os.makedirs(root, exist_ok=True)
    
    if not os.path.exists(manifest_path):
        # check if zip exists and is valid (handles manually downloaded files)
        zip_valid = False
        if os.path.exists(zip_path):
            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.testzip()
                zip_valid = True
                print(f"found existing zip file: {zip_path}")
                print("  if you manually downloaded this, it will be used.")
            except (zipfile.BadZipFile, zipfile.LargeZipFile) as e:
                print(f"existing zip file is corrupted: {e}")
                print("  please delete it and re-download or place a valid zip file.")
                response = input("  delete corrupted zip and try to download? (y/n): ")
                if response.lower() == 'y':
                    os.remove(zip_path)
                else:
                    raise ValueError("corrupted zip file must be removed or replaced")
        
        # download dataset if zip doesn't exist or is invalid
        if not zip_valid:
            print(f"\nno valid dataset found in {root}")
            print("options:")
            print("  1. manually download Dataset.zip (~9.3GB) and place it in:")
            print(f"     {os.path.abspath(root)}")
            print("  2. let the script download it automatically")
            print("\nchecking for manual download...")
            
            # wait a moment in case user just placed the file
            time.sleep(2)
            
            # check again after brief wait
            if os.path.exists(zip_path):
                try:
                    with zipfile.ZipFile(zip_path, 'r') as z:
                        z.testzip()
                    zip_valid = True
                    print(f"found manually placed zip file: {zip_path}")
                except (zipfile.BadZipFile, zipfile.LargeZipFile) as e:
                    print(f"manually placed zip file is corrupted: {e}")
                    raise ValueError("please provide a valid Dataset.zip file")
            
            # if still no valid zip, download
            if not zip_valid:
                print("\ndownloading dataset (this may take a while, ~9.3GB)...")
                print("  estimated time:")
                print("    - fast connection (100 mbps): ~15-20 minutes")
                print("    - medium connection (50 mbps): ~30-40 minutes")
                print("    - slow connection (10 mbps): ~2+ hours")
                print("  you can interrupt and resume later - the zip file will be preserved.")
                print("  for quick testing, use: python quick_preprocess.py\n")
                try:
                    af.dataset.download(root=root)
                except KeyboardInterrupt:
                    print("\ndownload interrupted. partial download saved.")
                    print("  run again to resume download.")
                    raise
                except Exception as e:
                    print(f"\ndownload error: {e}")
                    print("  you can try again - download should resume.")
                    print("  or manually download Dataset.zip and place it in:")
                    print(f"  {os.path.abspath(root)}")
                    raise
        
        # extract if zip exists but manifest doesn't
        if os.path.exists(zip_path) and not os.path.exists(manifest_path):
            print(f"\nextracting dataset from {zip_path} (this may take several minutes)...")
            print("  if you manually extracted the zip, make sure manifest.json is in:")
            print(f"  {os.path.abspath(root)}")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(root)
                print("extraction complete")
            except zipfile.BadZipFile:
                print("error: zip file is corrupted.")
                print("  if you manually downloaded this, please re-download.")
                print("  or delete the zip and let the script download it.")
                raise
    
    # check for manifest in subdirectories if not in root (recursive search)
    if not os.path.exists(manifest_path):
        # look for manifest in subdirectories (handles manual extraction)
        if os.path.exists(root) and os.path.isdir(root):
            # recursive search for manifest.json
            for dirpath, dirnames, filenames in os.walk(root):
                if 'manifest.json' in filenames:
                    manifest_path = os.path.join(dirpath, 'manifest.json')
                    root = dirpath
                    print(f"found manifest.json in subdirectory: {root}")
                    break
    
    # verify manifest exists before loading
    if not os.path.exists(manifest_path):
        error_msg = (
            f"manifest.json not found in {os.path.abspath(root)}.\n"
            f"please ensure the dataset was downloaded and extracted correctly.\n\n"
            f"if you manually downloaded Dataset.zip:\n"
            f"  1. place it in: {os.path.abspath(root)}\n"
            f"  2. extract it (or let the script extract it)\n"
            f"  3. make sure manifest.json is in the extracted folder\n\n"
            f"or delete any corrupted files and let the script download automatically."
        )
        raise FileNotFoundError(error_msg)
    
    # load dataset using airfrans api
    train = (split == 'train')
    dataset = af.dataset.load(root=root, task=task, train=train)
    
    if n_samples:
        return dataset[:n_samples]
    return dataset

def create_graph_data(sample, k=6):
    """convert airfrans point cloud sample to pytorch geometric data object"""
    # airfrans provides point clouds with features: position, inlet velocity, distance to airfoil
    # sample is a Simulation object or dict-like
    
    if hasattr(sample, 'pos'):
        pos = sample.pos
        inlet_velocity = sample.inlet_velocity if hasattr(sample, 'inlet_velocity') else None
        distance_to_airfoil = sample.distance_to_airfoil if hasattr(sample, 'distance_to_airfoil') else None
        pressure = sample.pressure if hasattr(sample, 'pressure') else None
        angle_of_attack = getattr(sample, 'angle_of_attack', 0.0)
        reynolds_number = getattr(sample, 'reynolds_number', 1e6)
    else:
        # handle dict-like access
        pos = sample.get('pos', sample.get('mesh_pos', None))
        inlet_velocity = sample.get('inlet_velocity', sample.get('velocity', None))
        distance_to_airfoil = sample.get('distance_to_airfoil', None)
        pressure = sample.get('pressure', None)
        angle_of_attack = sample.get('angle_of_attack', 0.0)
        reynolds_number = sample.get('reynolds_number', 1e6)
    
    # convert to tensors
    pos = torch.tensor(pos, dtype=torch.float)
    
    # build node features: position, inlet velocity, distance to airfoil
    node_features = [pos]
    
    if inlet_velocity is not None:
        inlet_velocity = torch.tensor(inlet_velocity, dtype=torch.float)
        if inlet_velocity.dim() == 1:
            inlet_velocity = inlet_velocity.unsqueeze(1)
        node_features.append(inlet_velocity)
    
    if distance_to_airfoil is not None:
        distance_to_airfoil = torch.tensor(distance_to_airfoil, dtype=torch.float)
        if distance_to_airfoil.dim() == 0:
            distance_to_airfoil = distance_to_airfoil.unsqueeze(0).expand(pos.size(0))
        if distance_to_airfoil.dim() == 1:
            distance_to_airfoil = distance_to_airfoil.unsqueeze(1)
        node_features.append(distance_to_airfoil)
    
    x = torch.cat(node_features, dim=1)
    
    # build edges using k-nn graph
    edge_index = knn_graph(pos, k=k, loop=False)
    
    # target: pressure coefficient (Cp) or pressure
    if pressure is not None:
        y = torch.tensor(pressure, dtype=torch.float)
        if y.dim() == 0:
            y = y.unsqueeze(0).expand(pos.size(0))
        if y.dim() == 1:
            y = y.unsqueeze(1)
    else:
        # if no pressure, use zeros (will need to be filled from dataset)
        y = torch.zeros(pos.size(0), 1, dtype=torch.float)
    
    # graph-level features: angle of attack, reynolds number
    graph_features = torch.tensor([
        angle_of_attack,
        reynolds_number
    ], dtype=torch.float)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        pos=pos,
        graph_features=graph_features
    )
    
    return data

def normalize_data(dataset):
    """normalize node features and targets"""
    all_x = []
    all_y = []
    
    for data in dataset:
        all_x.append(data.x)
        all_y.append(data.y)
    
    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)
    
    x_mean = all_x.mean(dim=0, keepdim=True)
    x_std = all_x.std(dim=0, keepdim=True) + 1e-8
    
    y_mean = all_y.mean()
    y_std = all_y.std() + 1e-8
    
    normalized_dataset = []
    for data in dataset:
        normalized_x = (data.x - x_mean) / x_std
        normalized_y = (data.y - y_mean) / y_std
        
        data.x = normalized_x
        data.y = normalized_y
        
        normalized_dataset.append(data)
    
    return normalized_dataset, {
        'x_mean': x_mean,
        'x_std': x_std,
        'y_mean': y_mean,
        'y_std': y_std
    }

def visualize_mesh(data, save_path='mesh_visualization.png'):
    """visualize airfoil mesh"""
    pos = data.pos.numpy()
    edge_index = data.edge_index.numpy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # plot edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        ax.plot([pos[src, 0], pos[dst, 0]], 
                [pos[src, 1], pos[dst, 1]], 
                'b-', alpha=0.3, linewidth=0.5)
    
    # plot nodes
    ax.scatter(pos[:, 0], pos[:, 1], c='red', s=1, alpha=0.6)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('airfoil mesh')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"mesh visualization saved to {save_path}")

def visualize_cp_heatmap(data, save_path='cp_heatmap.png'):
    """visualize pressure coefficient heatmap"""
    pos = data.pos.numpy()
    cp = data.y.numpy()
    
    # handle multi-dimensional y
    if cp.ndim > 1:
        cp = cp.squeeze()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(pos[:, 0], pos[:, 1], c=cp, 
                        cmap='coolwarm', s=2, alpha=0.8)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('pressure coefficient (Cp) distribution')
    ax.set_aspect('equal')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cp')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"cp heatmap saved to {save_path}")

def preprocess_dataset(split='train', n_samples=100, save_dir='./data', task='full'):
    """main preprocessing function"""
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"loading {split} dataset...")
    dataset = load_airfrans_data(split=split, n_samples=n_samples, task=task)
    
    # get dataset length
    try:
        dataset_len = len(dataset)
    except:
        dataset_len = n_samples if n_samples else 100
    
    print(f"converting to graph format...")
    graph_dataset = []
    for i, sample in enumerate(dataset):
        if i % 10 == 0:
            print(f"processing sample {i}/{dataset_len}")
        try:
            graph_data = create_graph_data(sample)
            graph_dataset.append(graph_data)
        except Exception as e:
            print(f"error processing sample {i}: {e}")
            continue
    
    if len(graph_dataset) == 0:
        raise ValueError("no valid samples processed")
    
    print("normalizing data...")
    normalized_dataset, norm_stats = normalize_data(graph_dataset)
    
    # visualize first sample
    print("creating visualizations...")
    visualize_mesh(normalized_dataset[0], 
                   save_path=os.path.join(save_dir, 'mesh_visualization.png'))
    visualize_cp_heatmap(normalized_dataset[0], 
                        save_path=os.path.join(save_dir, 'cp_heatmap.png'))
    
    # save dataset and normalization stats
    torch.save(normalized_dataset, os.path.join(save_dir, f'{split}_dataset.pt'))
    torch.save(norm_stats, os.path.join(save_dir, 'norm_stats.pt'))
    
    print(f"preprocessing complete. dataset saved to {save_dir}")
    return normalized_dataset, norm_stats

if __name__ == '__main__':
    import sys
    
    # check if user wants quick mode
    quick_mode = '--quick' in sys.argv or '-q' in sys.argv
    
    if quick_mode:
        print("=" * 60)
        print("QUICK MODE - Using smaller dataset for faster processing")
        print("=" * 60)
        print("For full dataset, run without --quick flag\n")
        train_samples = 50
        val_samples = 20
        test_samples = 20
    else:
        print("=" * 60)
        print("FULL MODE - Using complete dataset")
        print("=" * 60)
        print("For faster testing, use: python data_preprocess.py --quick\n")
        train_samples = 500
        val_samples = 100
        test_samples = 100
    
    # check if preprocessed data already exists
    if os.path.exists('./data/train_dataset.pt') and not quick_mode:
        print("preprocessed data already exists in ./data/")
        print("Delete ./data/ folder if you want to regenerate.\n")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Use existing data or delete ./data/ to regenerate.")
            sys.exit(0)
    
    # preprocess training data
    print(f"Preprocessing training data ({train_samples} samples)...")
    train_dataset, norm_stats = preprocess_dataset(split='train', n_samples=train_samples)
    
    # preprocess validation data
    print(f"Preprocessing validation data ({val_samples} samples)...")
    val_dataset, _ = preprocess_dataset(split='val', n_samples=val_samples)
    
    # preprocess test data
    print(f"Preprocessing test data ({test_samples} samples)...")
    test_dataset, _ = preprocess_dataset(split='test', n_samples=test_samples)
    
    print("\n" + "=" * 60)
    print("preprocessing complete!")
    print("=" * 60)

