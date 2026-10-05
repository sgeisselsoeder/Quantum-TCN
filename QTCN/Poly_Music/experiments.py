import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
import os
from collections import defaultdict
import argparse

import QTCN 
from utils import *

class PolyphonicMusicDataset(Dataset):
    
    def __init__(self, sequences, seq_length=32, reduce_features=True, target_features=2):
        self.sequences = []
        self.seq_length = seq_length
        self.reduce_features = reduce_features
        self.target_features = target_features
        for seq in sequences:
            if len(seq) > seq_length:
                for i in range(len(seq) - seq_length):
                    input_seq = seq[i:i+seq_length]
                    target = seq[i+seq_length]  
                    
                    if self.reduce_features and input_seq.shape[1] > self.target_features:
                        input_seq = self._reduce_dimensions(input_seq)
                        target = self._reduce_dimensions(target.unsqueeze(0)).squeeze(0)
                    
                    self.sequences.append((input_seq, target))
    
    def _reduce_dimensions(self, data):
        if data.shape[-1] <= self.target_features:
            return data
        if len(data.shape) == 2:  # sequence data
            variances = torch.var(data, dim=0)
            top_indices = torch.topk(variances, self.target_features).indices
            return data[:, top_indices]
        else:  
            variances = torch.var(data.unsqueeze(0), dim=0)
            top_indices = torch.topk(variances, self.target_features).indices
            return data[top_indices]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq, target = self.sequences[idx]
        return input_seq.float(), target.float()




def nll_loss(output, target, eps=1e-8):
    
    #sigmoid 
    output = torch.sigmoid(output)
    output = torch.clamp(output, eps, 1 - eps)
    
    #the NLL loss
    loss = -torch.trace(
        torch.matmul(target, torch.log(output).t()) +
        torch.matmul((1 - target), torch.log(1 - output).t())
    )
    
    return loss


def calculate_metrics(model, dataloader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    total_count = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            for i in range(inputs.size(0)):
                x = inputs[i]  
                y = targets[i]  
                output = model(x.unsqueeze(0)).squeeze(0)  
                loss = nll_loss(output.unsqueeze(0), y.unsqueeze(0))
                total_loss += loss.item()
                total_count += output.size(0)
                predicted = (torch.sigmoid(output) > 0.5).float()
                correct_predictions += (predicted == y).sum().item()
                total_predictions += y.numel()
    
    avg_loss = total_loss / total_count if total_count > 0 else 0.0
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    return avg_loss, accuracy



def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        if outputs.dim() > 1 and outputs.size(1) > 1:
            if targets.dim() == 2 and targets.size(1) != outputs.size(1):
                if targets.size(1) > outputs.size(1):
                    targets = targets[:, :outputs.size(1)]
                else:
                    pad_size = outputs.size(1) - targets.size(1)
                    targets = torch.cat([targets, torch.zeros(targets.size(0), pad_size, device=device)], dim=1)
        else:
            if targets.dim() > 1:
                targets = targets.mean(dim=1, keepdim=True)
        loss = criterion(outputs, targets)
        
        
        loss.backward()
        
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
    
    return total_loss / num_batches


def run_experiment(dataset_name, seq_length=32, hidden_dim=6, batch_size=32, 
                   num_epochs=100, learning_rate=0.001, device=None):
    
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running experiment on {dataset_name} dataset")
    print(f"Device: {device}")
    print(f"Hyperparameters:")
    print(f"  - Sequence length: {seq_length}")
    print(f"  - Hidden dimension: {hidden_dim}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Epochs: {num_epochs}")
    print("-" * 50)
    try:
        X_train, X_valid, X_test = data_generator(dataset_name)
        print(f"Data loaded successfully!")
        print(f"Train sequences: {len(X_train)}")
        print(f"Valid sequences: {len(X_valid)}")
        print(f"Test sequences: {len(X_test)}")
        original_features = X_train[0].shape[1] if len(X_train) > 0 else 2
        print(f"Original input features: {original_features}")
        target_features = 2
        #print(f"Reducing to {target_features} features to match model architecture")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure the data files are in './mdata/' directory")
        return None
    
   
    train_dataset = PolyphonicMusicDataset(X_train, seq_length, reduce_features=True, target_features=target_features)
    valid_dataset = PolyphonicMusicDataset(X_valid, seq_length, reduce_features=True, target_features=target_features)
    test_dataset = PolyphonicMusicDataset(X_test, seq_length, reduce_features=True, target_features=target_features)
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    model = QTCN.QTCN(
        seq_length=seq_length,
        input_features=target_features,  
        hidden_dim=hidden_dim,
        output_dim=target_features       
    ).to(device)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Model expects {target_features} input features")
    
    
    criterion = nn.BCEWithLogitsLoss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    
    history = {
        'train_loss': [],
        'valid_loss': [],
        'valid_accuracy': [],
        'test_loss': [],          
        'test_accuracy': [],      
        'learning_rate': []
    }
    
    best_valid_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    print("Starting training...")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss, valid_accuracy = calculate_metrics(model, valid_loader, device, criterion)
        test_loss, test_accuracy = calculate_metrics(model, test_loader, device, criterion)
        scheduler.step(valid_loss)
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['valid_accuracy'].append(valid_accuracy)
        history['test_loss'].append(test_loss)          
        history['test_accuracy'].append(test_accuracy) 
        history['learning_rate'].append(current_lr)
        
        epoch_time = time.time() - epoch_start_time
        
        # UPDATED: Print test metrics too
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Valid Loss: {valid_loss:.6f}")
        print(f"  Valid Accuracy: {valid_accuracy:.4f}")
        print(f"  Test Loss: {test_loss:.6f}")          
        print(f"  Test Accuracy: {test_accuracy:.4f}")   
        print(f"  Learning Rate: {current_lr:.2e}")
        print(f"  Time: {epoch_time:.2f}s")
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_qtcn_{dataset_name.lower()}.pth')
            print(f"  → New best model saved!")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        print("-" * 40)
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    model.load_state_dict(torch.load(f'best_qtcn_{dataset_name.lower()}.pth'))
    print("\nFinal Evaluation:")
    final_test_loss, final_test_accuracy = calculate_metrics(model, test_loader, device, criterion)
    print(f"Final Test Loss: {final_test_loss:.6f}")
    print(f"Final Test Accuracy: {final_test_accuracy:.4f}")
    test_losses_file = f'test_losses_{dataset_name.lower()}.txt'
    with open(test_losses_file, 'w') as f:
        f.write("# Test losses per epoch for plotting\n")
        f.write("# Format: epoch_number test_loss\n")
        for epoch, loss in enumerate(history['test_loss'], 1):
            f.write(f"{epoch} {loss:.8f}\n")
    print(f"Test losses saved to: {test_losses_file}")
    history_file = f'training_history_{dataset_name.lower()}.npz'
    np.savez(history_file, 
             train_loss=np.array(history['train_loss']),
             valid_loss=np.array(history['valid_loss']),
             valid_accuracy=np.array(history['valid_accuracy']),
             test_loss=np.array(history['test_loss']),
             test_accuracy=np.array(history['test_accuracy']),
             learning_rate=np.array(history['learning_rate']))
    print(f"Complete training history saved to: {history_file}")
    plot_training_history(history, dataset_name)
    
    return {
        'model': model,
        'history': history,
        'test_loss': final_test_loss,
        'test_accuracy': final_test_accuracy,
        'best_valid_loss': best_valid_loss
    }


def plot_training_history(history, dataset_name):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['valid_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.plot(epochs, history['test_loss'], 'g-', label='Test Loss', linewidth=2)  # NEW
    ax1.set_title('Training, Validation, and Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, history['valid_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.plot(epochs, history['test_accuracy'], 'g-', label='Test Accuracy', linewidth=2)  # NEW
    ax2.set_title('Validation and Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax3.semilogy(epochs, history['learning_rate'], 'purple', linewidth=2)
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate (log scale)')
    ax3.grid(True, alpha=0.3)
    if len(epochs) > 10:
        ax4.plot(epochs[10:], history['train_loss'][10:], 'b-', label='Training Loss', linewidth=2)
        ax4.plot(epochs[10:], history['valid_loss'][10:], 'r-', label='Validation Loss', linewidth=2)
        ax4.plot(epochs[10:], history['test_loss'][10:], 'g-', label='Test Loss', linewidth=2)  # NEW
        ax4.set_title('Loss Curves (After Epoch 10)')
    else:
        ax4.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax4.plot(epochs, history['valid_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax4.plot(epochs, history['test_loss'], 'g-', label='Test Loss', linewidth=2)  # NEW
        ax4.set_title('All Loss Curves')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'QTCN Training Results - {dataset_name} Dataset', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'qtcn_training_{dataset_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()




def plot_saved_test_losses(dataset_names, title="Test Loss Comparison"):
    
    plt.figure(figsize=(12, 8))
    
    for dataset in dataset_names:
        filename = f'test_losses_{dataset.lower()}.txt'
        try:
            epochs, losses = [], []
            with open(filename, 'r') as f:
                for line in f:
                    if not line.startswith('#'):
                        epoch, loss = line.strip().split()
                        epochs.append(int(epoch))
                        losses.append(float(loss))
            
            plt.plot(epochs, losses, linewidth=2, label=f'{dataset.upper()} Dataset')
            print(f"Loaded {len(losses)} test loss values for {dataset} dataset")
            
        except FileNotFoundError:
            print(f"File {filename} not found. Make sure you've run the experiment for {dataset} dataset.")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    datasets = ['Nott', 'JSB']
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"experiment with {dataset} dataset")
        print(f"{'='*80}\n")
        
        results = run_experiment(
            dataset_name=dataset,
            seq_length=32,
            hidden_dim=6,
            batch_size=32,
            num_epochs=100,
            learning_rate=0.001,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        if results:
            print(f"\n{dataset} experiment completed successfully!")
            print(f"Final test accuracy: {results['test_accuracy']:.4f}")
            print(f"Best validation loss: {results['best_valid_loss']:.6f}")
        else:
            print(f"\n{dataset} experiment failed")
        
        print(f"\n{'='*80}")
        print(f"{dataset} dataset experiment completed ")
        print(f"{'='*80}\n")
    
    plot_saved_test_losses(['nott', 'jsb'], "Test Loss Comparison: Nottingham vs JSB")