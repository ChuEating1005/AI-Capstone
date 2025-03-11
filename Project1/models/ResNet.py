import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError

class ImageFolderWithSkip(datasets.ImageFolder):
    def __getitem__(self, index):
        """
        Override the __getitem__ method to skip corrupted images
        """
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return sample, target
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Return a random valid image instead
            valid_idx = (index + 1) % len(self)
            return self.__getitem__(valid_idx)

class ResNetTrainer:
    def __init__(self, data_dir='dataset', batch_size=32, num_epochs=20, learning_rate=0.001):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Define transforms
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        
        # Check for corrupted images before loading datasets
        self._check_and_remove_corrupted_images(data_dir)
        
        # Load datasets with custom class that handles corrupted images
        self.image_datasets = {x: ImageFolderWithSkip(os.path.join(data_dir, x), 
                                                     self.data_transforms[x])
                              for x in ['train', 'valid', 'test']}
        
        # Create dataloaders
        self.dataloaders = {x: DataLoader(self.image_datasets[x], batch_size=batch_size,
                                         shuffle=True if x == 'train' else False, num_workers=4)
                           for x in ['train', 'valid', 'test']}
        
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'valid', 'test']}
        self.class_names = self.image_datasets['train'].classes
        
        print(f"Classes: {self.class_names}")
        print(f"Dataset sizes: {self.dataset_sizes}")
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.1)
        
        # For tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def _check_and_remove_corrupted_images(self, data_dir):
        """Check for and remove corrupted images in the dataset"""
        print("Checking for corrupted images...")
        corrupted_files = []
        
        for split in ['train', 'valid', 'test']:
            split_dir = os.path.join(data_dir, split)
            if not os.path.exists(split_dir):
                continue
                
            for root, _, files in os.walk(split_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(root, file)
                        try:
                            with Image.open(file_path) as img:
                                # Try to load the image
                                img.verify()
                        except Exception as e:
                            print(f"Corrupted image found: {file_path}")
                            print(f"Error: {e}")
                            corrupted_files.append(file_path)
        
        # Remove corrupted files
        for file_path in corrupted_files:
            try:
                os.remove(file_path)
                print(f"Removed corrupted file: {file_path}")
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}")
        
        if corrupted_files:
            print(f"Removed {len(corrupted_files)} corrupted images")
        else:
            print("No corrupted images found")
    
    def _initialize_model(self):
        # Load pre-trained ResNet-18 model
        model = models.resnet18(weights='IMAGENET1K_V1')
        
        # Modify the final fully connected layer for our number of classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(self.class_names))
        
        # Move model to device
        model = model.to(self.device)
        return model
    
    def train(self):
        best_val_acc = 0.0
        
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch+1}/{self.num_epochs}')
            print('-' * 10)
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
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
                else:
                    self.val_losses.append(epoch_loss)
                    self.val_accs.append(epoch_acc.item())
                    # Update learning rate scheduler
                    self.scheduler.step(epoch_loss)
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                # Save the best model
                if phase == 'valid' and epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    torch.save(self.model.state_dict(), 'result/resnet_best_model.pth')
            
            print()
        
        print(f'Best valid Acc: {best_val_acc:4f}')
        
        # Load best model weights
        self.model.load_state_dict(torch.load('result/resnet_best_model.pth'))
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
        plt.savefig('resnet_confusion_matrix.png')
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=self.class_names))
        
        return accuracy, cm
    
    def plot_training_history(self):
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Training Accuracy')
        plt.plot(self.val_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('resnet_training_history.png')
        plt.show()

if __name__ == "__main__":
    trainer = ResNetTrainer(data_dir='dataset', batch_size=32, num_epochs=20)
    trainer.train()
    trainer.evaluate()
    trainer.plot_training_history() 