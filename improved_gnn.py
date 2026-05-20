import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import os

class GATModel(nn.Module):
    """graph attention network with graph-level features"""
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, 
                 num_heads=4, dropout=0.2):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=dropout)
        
        # graph feature processing
        self.graph_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # combine node and graph features
        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
    
    def forward(self, x, edge_index, graph_features, batch):
        # graph convolutions
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # process graph features and broadcast to nodes
        graph_emb = self.graph_mlp(graph_features)
        
        # get unique batch indices
        unique_batches = torch.unique(batch)
        node_graph_emb = torch.zeros_like(x)
        for b in unique_batches:
            mask = (batch == b)
            node_graph_emb[mask] = graph_emb[b]
        
        # combine node and graph features
        x = torch.cat([x, node_graph_emb], dim=1)
        x = self.combine(x)
        x = F.relu(x)
        x = self.linear(x)
        
        return x

class GraphSAGEModel(nn.Module):
    """graphsage with graph-level features"""
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, dropout=0.2):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        
        # graph feature processing
        self.graph_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # combine node and graph features
        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
    
    def forward(self, x, edge_index, graph_features, batch):
        # graph convolutions
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # process graph features and broadcast to nodes
        graph_emb = self.graph_mlp(graph_features)
        
        # get unique batch indices
        unique_batches = torch.unique(batch)
        node_graph_emb = torch.zeros_like(x)
        for b in unique_batches:
            mask = (batch == b)
            node_graph_emb[mask] = graph_emb[b]
        
        # combine node and graph features
        x = torch.cat([x, node_graph_emb], dim=1)
        x = self.combine(x)
        x = F.relu(x)
        x = self.linear(x)
        
        return x

def compute_lift_coefficient(cp, pos, chord_length=1.0):
    """compute lift coefficient from pressure coefficient"""
    # simplified: integrate cp over surface
    # cl = integral of cp * n_y over surface
    # for now, use simplified approximation
    if len(cp.shape) > 1:
        cp = cp.squeeze()
    
    # find surface nodes (assuming y=0 or near y=0)
    surface_mask = np.abs(pos[:, 1]) < 0.01
    if np.sum(surface_mask) == 0:
        return 0.0
    
    surface_cp = cp[surface_mask]
    surface_x = pos[surface_mask, 0]
    
    # sort by x coordinate
    sort_idx = np.argsort(surface_x)
    surface_cp = surface_cp[sort_idx]
    surface_x = surface_x[sort_idx]
    
    # approximate cl using trapezoidal integration
    if len(surface_cp) < 2:
        return 0.0
    
    # simplified: cl = -integral of cp dx
    cl = -np.trapz(surface_cp, surface_x)
    
    return cl

def evaluate_with_lift(model, loader, criterion, device, norm_stats):
    """evaluate model and compute lift coefficients"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_cl_pred = []
    all_cl_true = []
    all_aoa = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # create batch tensor
            batch_tensor = torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
            if hasattr(batch, 'batch'):
                batch_tensor = batch.batch
            else:
                # create batch tensor manually
                idx = 0
                for i, data in enumerate(loader.dataset):
                    if idx >= batch.x.size(0):
                        break
                    num_nodes = data.x.size(0)
                    batch_tensor[idx:idx+num_nodes] = i
                    idx += num_nodes
            
            out = model(batch.x, batch.edge_index, 
                       batch.graph_features, batch_tensor)
            loss = criterion(out.squeeze(), batch.y)
            total_loss += loss.item()
            
            # denormalize predictions
            preds_denorm = out.squeeze().cpu().numpy() * norm_stats['y_std'].item() + norm_stats['y_mean'].item()
            targets_denorm = batch.y.cpu().numpy() * norm_stats['y_std'].item() + norm_stats['y_mean'].item()
            
            all_preds.append(out.squeeze().cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
            
            # compute lift coefficients for each sample in batch
            pos_denorm = batch.pos.cpu().numpy()
            for i in range(batch.num_graphs if hasattr(batch, 'num_graphs') else 1):
                # extract nodes for this graph
                start_idx = i * (batch.x.size(0) // (batch.num_graphs if hasattr(batch, 'num_graphs') else 1))
                end_idx = (i + 1) * (batch.x.size(0) // (batch.num_graphs if hasattr(batch, 'num_graphs') else 1))
                
                if end_idx > pos_denorm.shape[0]:
                    end_idx = pos_denorm.shape[0]
                
                sample_pos = pos_denorm[start_idx:end_idx]
                sample_cp_pred = preds_denorm[start_idx:end_idx]
                sample_cp_true = targets_denorm[start_idx:end_idx]
                
                cl_pred = compute_lift_coefficient(sample_cp_pred, sample_pos)
                cl_true = compute_lift_coefficient(sample_cp_true, sample_pos)
                
                all_cl_pred.append(cl_pred)
                all_cl_true.append(cl_true)
                
                if hasattr(batch, 'graph_features'):
                    aoa = batch.graph_features[i, 0].item()
                    all_aoa.append(aoa)
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    mse = mean_squared_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    # compute lift error
    cl_error = np.mean(np.abs(np.array(all_cl_pred) - np.array(all_cl_true))) / (np.abs(np.array(all_cl_true)).mean() + 1e-8) * 100
    
    return total_loss / len(loader), mse, r2, cl_error, all_preds, all_targets, all_cl_pred, all_cl_true, all_aoa

def plot_cp_contour_comparison(data, pred_cp, true_cp, save_path='cp_contour_comparison.png'):
    """plot comparison of predicted and true cp contours"""
    pos = data.pos.numpy()
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # true cp
    scatter1 = ax1.scatter(pos[:, 0], pos[:, 1], c=true_cp, 
                          cmap='coolwarm', s=2, alpha=0.8)
    ax1.set_title('true cp')
    ax1.set_aspect('equal')
    plt.colorbar(scatter1, ax=ax1)
    
    # predicted cp
    scatter2 = ax2.scatter(pos[:, 0], pos[:, 1], c=pred_cp, 
                          cmap='coolwarm', s=2, alpha=0.8)
    ax2.set_title('predicted cp')
    ax2.set_aspect('equal')
    plt.colorbar(scatter2, ax=ax2)
    
    # error
    error = pred_cp - true_cp
    scatter3 = ax3.scatter(pos[:, 0], pos[:, 1], c=error, 
                          cmap='RdBu', s=2, alpha=0.8)
    ax3.set_title('prediction error')
    ax3.set_aspect('equal')
    plt.colorbar(scatter3, ax=ax3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"cp contour comparison saved to {save_path}")

def plot_lift_vs_aoa(cl_pred, cl_true, aoa, save_path='lift_vs_aoa.png'):
    """plot lift coefficient vs angle of attack"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(aoa) > 0:
        aoa_array = np.array(aoa)
        cl_pred_array = np.array(cl_pred)
        cl_true_array = np.array(cl_true)
        
        # sort by aoa
        sort_idx = np.argsort(aoa_array)
        aoa_sorted = aoa_array[sort_idx]
        cl_pred_sorted = cl_pred_array[sort_idx]
        cl_true_sorted = cl_true_array[sort_idx]
        
        ax.plot(aoa_sorted, cl_true_sorted, 'o-', label='true cl', linewidth=2, markersize=6)
        ax.plot(aoa_sorted, cl_pred_sorted, 's-', label='predicted cl', linewidth=2, markersize=6)
    else:
        ax.plot(cl_true, 'o-', label='true cl', linewidth=2, markersize=6)
        ax.plot(cl_pred, 's-', label='predicted cl', linewidth=2, markersize=6)
    
    ax.set_xlabel('angle of attack (degrees)')
    ax.set_ylabel('lift coefficient (cl)')
    ax.set_title('lift coefficient vs angle of attack')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"lift vs aoa plot saved to {save_path}")

def train_improved_model(train_dataset, val_dataset, test_dataset, 
                        model_type='gat', epochs=100, batch_size=8, 
                        lr=0.001, hidden_dim=64, save_dir='./results_improved'):
    """train improved gnn model"""
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")
    
    # load normalization stats
    norm_stats = torch.load('./data/norm_stats.pt')
    
    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # get input dimension
    input_dim = train_dataset[0].x.shape[1]
    
    # create model
    if model_type == 'gat':
        model = GATModel(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    else:
        model = GraphSAGEModel(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"starting training with {model_type}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            batch_tensor = batch.batch if hasattr(batch, 'batch') else torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
            
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.graph_features, batch_tensor)
            loss = criterion(out.squeeze(), batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        val_loss, val_mse, val_r2, val_cl_error, _, _, _, _, _ = evaluate_with_lift(
            model, val_loader, criterion, device, norm_stats)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if epoch % 10 == 0:
            print(f"epoch {epoch}: train_loss={train_loss:.6f}, "
                  f"val_loss={val_loss:.6f}, val_mse={val_mse:.6f}, "
                  f"val_r2={val_r2:.4f}, val_cl_error={val_cl_error:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
    
    # evaluate on test set
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pt')))
    test_loss, test_mse, test_r2, test_cl_error, test_preds, test_targets, \
        test_cl_pred, test_cl_true, test_aoa = evaluate_with_lift(
            model, test_loader, criterion, device, norm_stats)
    
    print(f"\nfinal test results:")
    print(f"test_loss={test_loss:.6f}, test_mse={test_mse:.6f}, "
          f"test_r2={test_r2:.4f}, test_cl_error={test_cl_error:.2f}%")
    
    # create visualizations
    if len(test_dataset) > 0:
        sample_data = test_dataset[0]
        sample_pred = test_preds[:sample_data.x.size(0)]
        sample_true = test_targets[:sample_data.x.size(0)]
        
        # denormalize for visualization
        sample_pred_denorm = sample_pred * norm_stats['y_std'].item() + norm_stats['y_mean'].item()
        sample_true_denorm = sample_true * norm_stats['y_std'].item() + norm_stats['y_mean'].item()
        
        plot_cp_contour_comparison(sample_data, sample_pred_denorm, sample_true_denorm,
                                  save_path=os.path.join(save_dir, 'cp_contour_comparison.png'))
    
    if len(test_aoa) > 0:
        plot_lift_vs_aoa(test_cl_pred, test_cl_true, test_aoa,
                        save_path=os.path.join(save_dir, 'lift_vs_aoa.png'))
    
    # save results
    results = {
        'test_loss': test_loss,
        'test_mse': test_mse,
        'test_r2': test_r2,
        'test_cl_error': test_cl_error,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    torch.save(results, os.path.join(save_dir, 'results.pt'))
    
    return model, results

if __name__ == '__main__':
    # load datasets
    print("loading datasets...")
    train_dataset = torch.load('./data/train_dataset.pt')
    val_dataset = torch.load('./data/val_dataset.pt')
    test_dataset = torch.load('./data/test_dataset.pt')
    
    # train improved model (try both gat and sage)
    print("\n=== training GAT model ===")
    model_gat, results_gat = train_improved_model(
        train_dataset, val_dataset, test_dataset,
        model_type='gat', epochs=100, batch_size=8, lr=0.001, hidden_dim=64
    )
    
    print("\n=== training GraphSAGE model ===")
    model_sage, results_sage = train_improved_model(
        train_dataset, val_dataset, test_dataset,
        model_type='sage', epochs=100, batch_size=8, lr=0.001, hidden_dim=64,
        save_dir='./results_sage'
    )

