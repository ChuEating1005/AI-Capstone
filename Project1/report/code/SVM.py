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
from dataset import SVMDataset

class SVMClassifier:
    def __init__(self, data_dir='dataset'):
        self.data_dir = Path(data_dir)
        self.class_names = None
        self.model = None
        self.scaler = None
        self.dataset = SVMDataset(data_dir, 0)

    def train(self):
        """Train the SVM model with hyperparameter tuning"""
        # Load data
        X_train, y_train, X_test, y_test = self.dataset.load_data()
        
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
        
        # Check if class_names is populated
        if not self.class_names:
            self.class_names = [str(i) for i in range(len(set(y_test)))]
        
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