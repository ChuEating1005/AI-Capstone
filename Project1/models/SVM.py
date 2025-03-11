import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from PIL import Image
import joblib
from pathlib import Path

class SVMClassifier:
    def __init__(self, data_dir='dataset'):
        self.data_dir = Path(data_dir)
        self.class_names = None
        self.model = None
        self.scaler = None
    
    def extract_features(self, img_path):
        """
        Extract simple features from an image.
        For SVM, we'll use a combination of color histograms and HOG-like features.
        """
        try:
            # Load and resize image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((64, 64))  # Resize for consistency
            img_array = np.array(img)
            
            # Extract color histograms (simple feature)
            hist_r = np.histogram(img_array[:,:,0], bins=32, range=(0, 256))[0]
            hist_g = np.histogram(img_array[:,:,1], bins=32, range=(0, 256))[0]
            hist_b = np.histogram(img_array[:,:,2], bins=32, range=(0, 256))[0]
            
            # Simple edge detection (gradient-based features)
            gray = np.mean(img_array, axis=2).astype(np.uint8)
            grad_x = np.abs(np.gradient(gray, axis=1)).flatten()
            grad_y = np.abs(np.gradient(gray, axis=0)).flatten()
            
            # Downsample gradients to reduce dimensionality
            grad_x = np.histogram(grad_x, bins=32, range=(0, 256))[0]
            grad_y = np.histogram(grad_y, bins=32, range=(0, 256))[0]
            
            # Combine features
            features = np.concatenate([hist_r, hist_g, hist_b, grad_x, grad_y])
            return features
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None
    
    def load_data(self):
        """Load and prepare data for SVM training"""
        X_train, y_train = [], []
        X_val, y_val = [], []
        X_test, y_test = [], []
        
        # Get class names from train directory
        self.class_names = [d for d in os.listdir(self.data_dir / 'train') 
                           if os.path.isdir(self.data_dir / 'train' / d)]
        
        # Create class to index mapping
        class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        
        # Process training data
        print("Loading training data...")
        for class_name in self.class_names:
            class_dir = self.data_dir / 'train' / class_name
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = class_dir / img_file
                    features = self.extract_features(img_path)
                    if features is not None:
                        X_train.append(features)
                        y_train.append(class_to_idx[class_name])
        
        # Process validation data
        print("Loading validation data...")
        for class_name in self.class_names:
            class_dir = self.data_dir / 'valid' / class_name
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = class_dir / img_file
                    features = self.extract_features(img_path)
                    if features is not None:
                        X_val.append(features)
                        y_val.append(class_to_idx[class_name])
        
        # Process test data
        print("Loading test data...")
        for class_name in self.class_names:
            class_dir = self.data_dir / 'test' / class_name
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = class_dir / img_file
                    features = self.extract_features(img_path)
                    if features is not None:
                        X_test.append(features)
                        y_test.append(class_to_idx[class_name])
        
        return (np.array(X_train), np.array(y_train), 
                np.array(X_val), np.array(y_val),
                np.array(X_test), np.array(y_test))
    
    def train(self):
        """Train the SVM model with hyperparameter tuning"""
        # Load data
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data()
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        # Create a pipeline with scaling and SVM
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True))
        ])
        
        # Define parameter grid for grid search
        param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto', 0.01, 0.1],
            'svm__kernel': ['rbf', 'poly']
        }
        
        # Perform grid search with cross-validation
        print("Performing grid search for hyperparameter tuning...")
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        self.model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Evaluate on validation set
        val_preds = self.model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_preds)
        print(f"Validation accuracy: {val_accuracy:.4f}")
        
        # Save the model
        joblib.dump(self.model, 'svm_model.pkl')
        
        # Evaluate on test set
        test_preds = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_preds)
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, test_preds)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('SVM Confusion Matrix')
        plt.savefig('svm_confusion_matrix.png')
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, test_preds, target_names=self.class_names))
        
        return test_accuracy, cm

if __name__ == "__main__":
    classifier = SVMClassifier(data_dir='dataset')
    classifier.train() 