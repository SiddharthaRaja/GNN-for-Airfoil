import torch
from torch_geometric.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import os
from improved_gnn import GATModel, GraphSAGEModel, evaluate_with_lift, compute_lift_coefficient

def plot_cp_vs_chordwise(pos, cp_pred, cp_true, save_path='cp_vs_chordwise.png'):
    """plot cp vs chordwise position"""
    # extract upper and lower surfaces
    upper_mask = pos[:, 1] > 0
    lower_mask = pos[:, 1] < 0
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # upper surface
    if np.sum(upper_mask) > 0:
        upper_x = pos[upper_mask, 0]
        upper_cp_pred = cp_pred[upper_mask]
        upper_cp_true = cp_true[upper_mask]
        
        # sort by x
        sort_idx = np.argsort(upper_x)
        ax1.plot(upper_x[sort_idx], upper_cp_true[sort_idx], 'o-', 
                label='true', linewidth=2, markersize=4, alpha=0.7)
        ax1.plot(upper_x[sort_idx], upper_cp_pred[sort_idx], 's-', 
                label='predicted', linewidth=2, markersize=4, alpha=0.7)
    
    ax1.set_xlabel('chordwise position (x/c)')
    ax1.set_ylabel('cp')
    ax1.set_title('upper surface')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    
    # lower surface
    if np.sum(lower_mask) > 0:
        lower_x = pos[lower_mask, 0]
        lower_cp_pred = cp_pred[lower_mask]
        lower_cp_true = cp_true[lower_mask]
        
        # sort by x
        sort_idx = np.argsort(lower_x)
        ax2.plot(lower_x[sort_idx], lower_cp_true[sort_idx], 'o-', 
                label='true', linewidth=2, markersize=4, alpha=0.7)
        ax2.plot(lower_x[sort_idx], lower_cp_pred[sort_idx], 's-', 
                label='predicted', linewidth=2, markersize=4, alpha=0.7)
    
    ax2.set_xlabel('chordwise position (x/c)')
    ax2.set_ylabel('cp')
    ax2.set_title('lower surface')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"cp vs chordwise plot saved to {save_path}")

def plot_scatter_comparison(preds, targets, save_path='scatter_comparison.png'):
    """plot scatter comparison of predictions"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.scatter(targets, preds, alpha=0.3, s=1)
    
    # perfect prediction line
    min_val = min(targets.min(), preds.min())
    max_val = max(targets.max(), preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='perfect prediction')
    
    # compute metrics
    mse = mean_squared_error(targets, preds)
    r2 = r2_score(targets, preds)
    mae = np.mean(np.abs(preds - targets))
    
    ax.set_xlabel('true cp')
    ax.set_ylabel('predicted cp')
    ax.set_title(f'prediction scatter plot (mse={mse:.6f}, r²={r2:.4f}, mae={mae:.6f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"scatter comparison saved to {save_path}")

def evaluate_unseen_airfoils(model, test_dataset, norm_stats, model_type='gat', 
                            device='cuda', save_dir='./evaluation_results'):
    """evaluate on unseen airfoils"""
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    all_preds = []
    all_targets = []
    all_cl_pred = []
    all_cl_true = []
    all_aoa = []
    
    print("evaluating on unseen airfoils...")
    for idx, batch in enumerate(test_loader):
        batch = batch.to(device)
        batch_tensor = batch.batch if hasattr(batch, 'batch') else torch.zeros(batch.x.size(0), dtype=torch.long, device=device)
        
        with torch.no_grad():
            out = model(batch.x, batch.edge_index, batch.graph_features, batch_tensor)
        
        # denormalize
        pred_denorm = out.squeeze().cpu().numpy() * norm_stats['y_std'].item() + norm_stats['y_mean'].item()
        target_denorm = batch.y.cpu().numpy() * norm_stats['y_std'].item() + norm_stats['y_mean'].item()
        
        all_preds.append(pred_denorm)
        all_targets.append(target_denorm)
        
        # compute lift coefficients
        pos = batch.pos.cpu().numpy()
        cl_pred = compute_lift_coefficient(pred_denorm, pos)
        cl_true = compute_lift_coefficient(target_denorm, pos)
        
        all_cl_pred.append(cl_pred)
        all_cl_true.append(cl_true)
        
        if hasattr(batch, 'graph_features'):
            aoa = batch.graph_features[0, 0].item()
            all_aoa.append(aoa)
        
        # create individual visualizations for first few samples
        if idx < 5:
            plot_cp_vs_chordwise(pos, pred_denorm, target_denorm,
                                save_path=os.path.join(save_dir, f'cp_chordwise_sample_{idx}.png'))
    
    # concatenate all predictions
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # compute final metrics
    mse = mean_squared_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    mae = np.mean(np.abs(all_preds - all_targets))
    
    cl_error = np.mean(np.abs(np.array(all_cl_pred) - np.array(all_cl_true))) / (np.abs(np.array(all_cl_true)).mean() + 1e-8) * 100
    
    print(f"\nfinal evaluation metrics:")
    print(f"mse: {mse:.6f}")
    print(f"r²: {r2:.4f}")
    print(f"mae: {mae:.6f}")
    print(f"lift error: {cl_error:.2f}%")
    
    # create scatter plot
    plot_scatter_comparison(all_preds, all_targets,
                          save_path=os.path.join(save_dir, 'scatter_comparison.png'))
    
    # save results
    results = {
        'mse': mse,
        'r2': r2,
        'mae': mae,
        'cl_error': cl_error,
        'all_preds': all_preds,
        'all_targets': all_targets,
        'all_cl_pred': all_cl_pred,
        'all_cl_true': all_cl_true
    }
    
    torch.save(results, os.path.join(save_dir, 'evaluation_results.pt'))
    
    return results

if __name__ == '__main__':
    # load datasets and normalization stats
    print("loading datasets...")
    test_dataset = torch.load('./data/test_dataset.pt')
    norm_stats = torch.load('./data/norm_stats.pt')
    
    # load trained models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # evaluate GAT model
    print("\n=== evaluating GAT model ===")
    train_dataset = torch.load('./data/train_dataset.pt')
    input_dim = train_dataset[0].x.shape[1]
    model_gat = GATModel(input_dim=input_dim, hidden_dim=64).to(device)
    model_gat.load_state_dict(torch.load('./results_improved/best_model.pt'))
    
    results_gat = evaluate_unseen_airfoils(
        model_gat, test_dataset, norm_stats, model_type='gat',
        save_dir='./evaluation_results_gat'
    )
    
    # evaluate GraphSAGE model
    print("\n=== evaluating GraphSAGE model ===")
    model_sage = GraphSAGEModel(input_dim=input_dim, hidden_dim=64).to(device)
    model_sage.load_state_dict(torch.load('./results_sage/best_model.pt'))
    
    results_sage = evaluate_unseen_airfoils(
        model_sage, test_dataset, norm_stats, model_type='sage',
        save_dir='./evaluation_results_sage'
    )

