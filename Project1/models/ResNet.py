import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from torchvision import models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE
from dataset import ResnetDataset, train_tfm, valid_tfm, test_tfm

# Set random seeds for reproducibility
def set_seed(seed=40):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class ResNetTrainer:
    def __init__(self, data_dir='data', batch_size=32, num_epochs=40, learning_rate=0.001, seed=40):
        # Set seed for reproducibility
        set_seed(seed)
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")        
        
        # Load datasets with custom class that handles corrupted images
        self.datasets = {
            'train': ResnetDataset(os.path.join(data_dir, 'train'), transform=train_tfm),
            'test': ResnetDataset(os.path.join(data_dir, 'test'), transform=test_tfm)
        }
        
        # Create dataloaders
        self.dataloaders = {
            'train': DataLoader(self.datasets['train'], batch_size=batch_size, shuffle=True, num_workers=6),
            'test': DataLoader(self.datasets['test'], batch_size=batch_size, shuffle=False, num_workers=2)
        }
        
        self.dataset_sizes = {x: len(self.datasets[x]) for x in ['train', 'test']}
        self.class_names = self.datasets['train'].classes
        
        print(f"Classes: {self.class_names}")
        print(f"Dataset sizes: {self.dataset_sizes}")
        
        # Initialize model
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, len(self.class_names))
        )
        self.model = self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        # For tracking metrics
        self.train_losses = []
        self.train_accs = []
    
    def train(self):
        best_test_acc = 0.0
        
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch+1}/{self.num_epochs}')
            print('-' * 10)
            
            # Each epoch has a training and testing phase
            for phase in ['train', 'test']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode
                
                running_loss = 0.0
                running_corrects = 0
                
                # Wrap dataloader with tqdm for progress bar
                pbar = tqdm(self.dataloaders[phase], desc=f'{phase} Epoch {epoch+1}/{self.num_epochs}')
                
                # Iterate over data
                for inputs, labels in pbar:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Zero the parameter gradients
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        
                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    
                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                    # Update progress bar
                    batch_loss = loss.item()
                    batch_acc = torch.sum(preds == labels.data).double() / inputs.size(0)
                    pbar.set_postfix({'loss': f'{batch_loss:.4f}', 'acc': f'{batch_acc:.4f}'})
                
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                
                if phase == 'train':
                    self.train_losses.append(epoch_loss)
                    self.train_accs.append(epoch_acc.item())
                    self.scheduler.step()  # Step the scheduler
                else:
                    # Save the best model
                    if epoch_acc > best_test_acc:
                        best_test_acc = epoch_acc
                        torch.save(self.model.state_dict(), 'results/resnet_best_model.pth')
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            print()
        
        print(f'Best test Acc: {best_test_acc:4f}')
        
        # Load best model weights
        self.model.load_state_dict(torch.load('results/resnet_best_model.pth'))
        return self.model
    
    def evaluate(self):
        # Set model to evaluate mode
        self.model.eval()
        
        # Initialize lists to store predictions and ground truth
        all_preds = []
        all_labels = []
        
        # No gradient calculation needed
        with torch.no_grad():
            # Wrap dataloader with tqdm for progress bar
            pbar = tqdm(self.dataloaders['test'], desc='Evaluating')
            
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                # Store predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                batch_acc = torch.sum(preds == labels.data).double() / inputs.size(0)
                pbar.set_postfix({'batch_acc': f'{batch_acc:.4f}'})
        
        # Calculate accuracy
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        print(f'Test Accuracy: {accuracy:.4f}')
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('plots/resnet_confusion_matrix.png')
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=self.class_names))
        
        return accuracy, cm
    
    def plot_training_history(self):
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('plots/resnet_training_history.png')
        plt.show()

    
    def visualize_tsne_comparison(self, layers=['input', 'layer1', 'layer2', 'layer3', 'layer4', 'fc'], perplexity=30, n_iter=1000):
        """
        Visualize and compare feature representations from multiple layers using t-SNE

        Args:
            layers (list): List of layers to extract features from
            perplexity (int): Perplexity parameter for t-SNE
            n_iter (int): Number of iterations for t-SNE
        """
        self.model.eval()
        
        layer_indices = {
            'layer1': 4,
            'layer2': 5,
            'layer3': 6,
            'layer4': 7
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, layer in enumerate(layers):
            if layer == 'input':
                feature_extractor = torch.nn.Identity()
            elif layer in layer_indices:
                feature_extractor = torch.nn.Sequential(*list(self.model.children())[:layer_indices[layer]])
            elif layer == 'fc':
                feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
            else:
                raise ValueError(f"Layer {layer} not supported for feature extraction")
            
            features = []
            labels = []
            
            with torch.no_grad():
                for inputs, targets in tqdm(self.dataloaders['test'], desc=f'Extracting features from {layer}'):
                    inputs = inputs.to(self.device)
                    feat = feature_extractor(inputs)
                    feat = feat.view(feat.size(0), -1)
                    features.append(feat.cpu().numpy())
                    labels.append(targets.numpy())
            
            features = np.concatenate(features, axis=0)
            labels = np.concatenate(labels, axis=0)
            
            print(f"Applying t-SNE on {features.shape[0]} samples with {features.shape[1]} features from {layer}...")
            tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=40)
            features_tsne = tsne.fit_transform(features)
            
            ax = axes[i]
            for _, label in enumerate(np.unique(labels)):
                idx = labels == label
                ax.scatter(features_tsne[idx, 0], features_tsne[idx, 1], label=self.class_names[label], alpha=0.7, s=50)
            
            ax.set_title(f'{layer}', fontsize=18)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('plots/resnet_tsne_comparison.png', dpi=300)
        plt.show()


if __name__ == "__main__":
    trainer = ResNetTrainer(data_dir='data', batch_size=32, num_epochs=40, seed=40)
    trainer.train()
    trainer.evaluate()
    trainer.plot_training_history()
    trainer.visualize_tsne_comparison() 