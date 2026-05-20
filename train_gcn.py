import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import os

class GCNModel(nn.Module):
    """2-layer graph convolutional network"""
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, dropout=0.2):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.linear(x)
        return x

def train_epoch(model, loader, optimizer, criterion, device):
    """train for one epoch"""
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = criterion(out.squeeze(), batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    """evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            loss = criterion(out.squeeze(), batch.y)
            total_loss += loss.item()
            
            all_preds.append(out.squeeze().cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    mse = mean_squared_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    return total_loss / len(loader), mse, r2, all_preds, all_targets

def plot_training_history(train_losses, val_losses, save_path='training_history.png'):
    """plot training and validation losses"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label='train loss', alpha=0.7)
    ax.plot(val_losses, label='val loss', alpha=0.7)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title('training history')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"training history saved to {save_path}")

def plot_predictions(preds, targets, save_path='predictions.png'):
    """plot predicted vs true values"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # scatter plot
    ax1.scatter(targets, preds, alpha=0.3, s=1)
    ax1.plot([targets.min(), targets.max()], 
             [targets.min(), targets.max()], 'r--', lw=2)
    ax1.set_xlabel('true Cp')
    ax1.set_ylabel('predicted Cp')
    ax1.set_title('predicted vs true Cp')
    ax1.grid(True, alpha=0.3)
    
    # error distribution
    errors = preds - targets
    ax2.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('prediction error')
    ax2.set_ylabel('frequency')
    ax2.set_title('error distribution')
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"predictions plot saved to {save_path}")

def train_model(train_dataset, val_dataset, test_dataset, 
                epochs=100, batch_size=8, lr=0.001, 
                hidden_dim=64, save_dir='./results'):
    """main training function"""
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")
    
    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # get input dimension
    input_dim = train_dataset[0].x.shape[1]
    
    # create model
    model = GCNModel(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("starting training...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mse, val_r2, _, _ = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if epoch % 10 == 0:
            print(f"epoch {epoch}: train_loss={train_loss:.6f}, "
                  f"val_loss={val_loss:.6f}, val_mse={val_mse:.6f}, val_r2={val_r2:.4f}")
        
        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
    
    # load best model and evaluate on test set
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pt')))
    test_loss, test_mse, test_r2, test_preds, test_targets = evaluate(
        model, test_loader, criterion, device)
    
    print(f"\nfinal test results:")
    print(f"test_loss={test_loss:.6f}, test_mse={test_mse:.6f}, test_r2={test_r2:.4f}")
    
    # create plots
    plot_training_history(train_losses, val_losses, 
                         save_path=os.path.join(save_dir, 'training_history.png'))
    plot_predictions(test_preds, test_targets, 
                    save_path=os.path.join(save_dir, 'predictions.png'))
    
    # save results
    results = {
        'test_loss': test_loss,
        'test_mse': test_mse,
        'test_r2': test_r2,
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
    
    # train model
    model, results = train_model(
        train_dataset, val_dataset, test_dataset,
        epochs=100, batch_size=8, lr=0.001, hidden_dim=64
    )

