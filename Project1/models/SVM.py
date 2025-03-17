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
import cv2
from skimage.feature import local_binary_pattern, hog

class SVMClassifier:
    def __init__(self, data_dir='dataset'):
        self.data_dir = Path(data_dir)
        self.class_names = None
        self.model = None
        self.scaler = None
    
    def extract_features(self, img_path):
        """
        Enhanced feature extraction for SVM classification:
        1. HSV color features
        2. Texture features (LBP)
        3. Shape features (HOG)
        4. Edge features (Canny)
        """
        try:
            # Load and resize image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((128, 128))
            img_array = np.array(img)
            
            # 1. Color Features (HSV space)
            img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # HSV histograms
            hist_h = np.histogram(img_hsv[:,:,0], bins=16)[0]
            hist_s = np.histogram(img_hsv[:,:,1], bins=16)[0]
            hist_v = np.histogram(img_hsv[:,:,2], bins=16)[0]
            
            # Color statistics
            color_stats = np.concatenate([
                np.mean(img_hsv, axis=(0,1)),  # mean of each channel
                np.std(img_hsv, axis=(0,1)),   # std of each channel
                np.percentile(img_hsv, [25,75], axis=(0,1)).flatten()  # quartiles
            ])
            
            # 2. Texture Features
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Multi-scale LBP
            lbp_features = []
            for radius in [1, 2, 3]:
                lbp = local_binary_pattern(gray, P=8*radius, R=radius, method='uniform')
                lbp_hist = np.histogram(lbp, bins=10)[0]
                lbp_features.extend(lbp_hist/lbp_hist.sum())  # 正規化
            
            # 3. Shape Features (HOG)
            hog_features = hog(gray, 
                             orientations=9,
                             pixels_per_cell=(16, 16),
                             cells_per_block=(2, 2),
                             visualize=False)
            
            # 4. Edge Features
            edges = cv2.Canny(gray, 100, 200)
            edge_hist = np.histogram(edges, bins=16)[0]
            
            # Gradient direction histograms
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            grad_dir = np.arctan2(grad_y, grad_x)
            grad_hist = np.histogram(grad_dir, bins=18, range=(-np.pi, np.pi))[0]
            
            # 組合所有特徵並正規化
            features = np.concatenate([
                hist_h/hist_h.sum(),           # 16 features
                hist_s/hist_s.sum(),           # 16 features
                hist_v/hist_v.sum(),           # 16 features
                color_stats,                    # 12 features
                np.array(lbp_features),         # 30 features
                hog_features/np.linalg.norm(hog_features),  # normalized HOG
                edge_hist/edge_hist.sum(),      # 16 features
                grad_hist/grad_hist.sum()       # 18 features
            ])
            
            return features
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None
    
    def load_data(self):
        """Load and prepare data for SVM training"""
        X_train, y_train = [], []
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
                np.array(X_test), np.array(y_test))
    
    def train(self):
        """Train the SVM model with hyperparameter tuning"""
        # Load data
        X_train, y_train, X_test, y_test = self.load_data()
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        # Create a pipeline with scaling and SVM
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True))
        ])
        
        # Define parameter grid for grid search
        param_grid = {
            'svm__C': [0.1, 1, 10, 100, 1000],
            'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'svm__kernel': ['rbf', 'poly'],
            'svm__degree': [2, 3] # for poly kernel
        }
        
        # Perform grid search with cross-validation
        print("Performing grid search for hyperparameter tuning...")
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        
        # Get the best model
        self.model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
        # Save the model
        joblib.dump(self.model, 'results/svm_model.pkl')
        
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
        plt.savefig('plots/svm_confusion_matrix.png')
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, test_preds, target_names=self.class_names))
        
        return test_accuracy, cm

if __name__ == "__main__":
    classifier = SVMClassifier(data_dir='dataset')
    classifier.train() 