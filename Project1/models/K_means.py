import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
import seaborn as sns
from PIL import Image
import joblib
from pathlib import Path
from collections import Counter
import pandas as pd

class KMeansClusterer:
    def __init__(self, data_dir='dataset', n_clusters=10):
        self.data_dir = Path(data_dir)
        self.n_clusters = n_clusters
        self.model = None
        self.pca = None
        self.scaler = None
        self.class_names = None
    
    def extract_features(self, img_path):
        """
        Extract features from an image for clustering.
        Similar to the SVM feature extraction but can be simpler.
        """
        try:
            # Load and resize image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((64, 64))  # Resize for consistency
            img_array = np.array(img)
            
            # Extract color histograms
            hist_r = np.histogram(img_array[:,:,0], bins=32, range=(0, 256))[0]
            hist_g = np.histogram(img_array[:,:,1], bins=32, range=(0, 256))[0]
            hist_b = np.histogram(img_array[:,:,2], bins=32, range=(0, 256))[0]
            
            # Calculate mean and std of each channel
            mean_r = np.mean(img_array[:,:,0])
            mean_g = np.mean(img_array[:,:,1])
            mean_b = np.mean(img_array[:,:,2])
            std_r = np.std(img_array[:,:,0])
            std_g = np.std(img_array[:,:,1])
            std_b = np.std(img_array[:,:,2])
            
            # Combine features
            features = np.concatenate([
                hist_r, hist_g, hist_b, 
                [mean_r, mean_g, mean_b, std_r, std_g, std_b]
            ])
            return features
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None
    
    def load_data(self):
        """Load and prepare data for K-means clustering"""
        X = []  # Features
        y = []  # True labels (for evaluation only)
        img_paths = []  # Store paths for visualization
        
        # Get class names from train directory
        self.class_names = [d for d in os.listdir(self.data_dir / 'train') 
                           if os.path.isdir(self.data_dir / 'train' / d)]
        
        # Create class to index mapping
        class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        
        # Process all data (train, valid, test combined for unsupervised learning)
        for split in ['train', 'valid', 'test']:
            for class_name in self.class_names:
                class_dir = self.data_dir / split / class_name
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = class_dir / img_file
                        features = self.extract_features(img_path)
                        if features is not None:
                            X.append(features)
                            y.append(class_to_idx[class_name])
                            img_paths.append(str(img_path))
        
        return np.array(X), np.array(y), img_paths
    
    def train(self):
        """Train the K-means clustering model"""
        # Load data
        X, y_true, img_paths = self.load_data()
        
        print(f"Data shape: {X.shape}")
        
        # Scale the features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction
        self.pca = PCA(n_components=50)  # Reduce to 50 dimensions
        X_pca = self.pca.fit_transform(X_scaled)
        
        print(f"PCA explained variance ratio: {np.sum(self.pca.explained_variance_ratio_):.4f}")
        
        # Find optimal number of clusters using silhouette score
        if self.n_clusters is None:
            silhouette_scores = []
            range_n_clusters = range(2, 20)  # Try from 2 to 20 clusters
            
            for n_clusters in range_n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_pca)
                silhouette_avg = silhouette_score(X_pca, cluster_labels)
                silhouette_scores.append(silhouette_avg)
                print(f"For n_clusters = {n_clusters}, silhouette score is {silhouette_avg:.4f}")
            
            # Plot silhouette scores
            plt.figure(figsize=(10, 6))
            plt.plot(range_n_clusters, silhouette_scores, 'o-')
            plt.xlabel('Number of clusters')
            plt.ylabel('Silhouette Score')
            plt.title('Silhouette Score Method For Optimal k')
            plt.savefig('kmeans_silhouette_scores.png')
            
            # Choose the best number of clusters
            self.n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
            print(f"Optimal number of clusters: {self.n_clusters}")
        
        # Train K-means with the optimal/specified number of clusters
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = self.model.fit_predict(X_pca)
        
        # Save the model
        joblib.dump((self.model, self.pca, self.scaler), 'kmeans_model.pkl')
        
        # Evaluate clustering performance
        if len(np.unique(y_true)) == self.n_clusters:
            ari = adjusted_rand_score(y_true, cluster_labels)
            print(f"Adjusted Rand Index: {ari:.4f}")
        
        # Visualize clusters in 2D
        self._visualize_clusters(X_pca, cluster_labels, y_true)
        
        # Analyze cluster composition
        self._analyze_clusters(cluster_labels, y_true)
        
        return self.model
    
    def _visualize_clusters(self, X_pca, cluster_labels, y_true):
        """Visualize the clusters in 2D using PCA"""
        # Further reduce to 2D for visualization
        pca_viz = PCA(n_components=2)
        X_pca_2d = pca_viz.fit_transform(X_pca)
        
        # Plot clusters
        plt.figure(figsize=(12, 10))
        
        # Plot by cluster assignment
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('K-means Clustering Results')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        
        # Plot by true labels
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y_true, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('True Class Labels')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        
        plt.tight_layout()
        plt.savefig('kmeans_clusters_visualization.png')
    
    def _analyze_clusters(self, cluster_labels, y_true):
        """Analyze the composition of each cluster"""
        # Create a mapping from cluster labels to true labels
        cluster_to_label = {}
        
        for cluster_id in range(self.n_clusters):
            # Get indices of samples in this cluster
            indices = np.where(cluster_labels == cluster_id)[0]
            
            # Get true labels of these samples
            true_labels = y_true[indices]
            
            # Count occurrences of each true label
            label_counts = Counter(true_labels)
            
            # Find the most common true label in this cluster
            most_common_label = label_counts.most_common(1)[0][0]
            cluster_to_label[cluster_id] = most_common_label
            
            # Print cluster composition
            print(f"\nCluster {cluster_id} composition:")
            for label, count in label_counts.most_common():
                percentage = count / len(indices) * 100
                print(f"  {self.class_names[label]}: {count} samples ({percentage:.2f}%)")
        
        # Create confusion matrix-like visualization
        confusion = np.zeros((self.n_clusters, len(self.class_names)))
        
        for i in range(len(cluster_labels)):
            cluster = cluster_labels[i]
            true_label = y_true[i]
            confusion[cluster, true_label] += 1
        
        # Normalize by cluster size
        for i in range(self.n_clusters):
            if np.sum(confusion[i, :]) > 0:
                confusion[i, :] = confusion[i, :] / np.sum(confusion[i, :])
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(confusion, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=[f'Cluster {i}' for i in range(self.n_clusters)])
        plt.xlabel('True Class')
        plt.ylabel('Cluster')
        plt.title('Cluster Composition')
        plt.tight_layout()
        plt.savefig('kmeans_cluster_composition.png')

if __name__ == "__main__":
    clusterer = KMeansClusterer(data_dir='dataset', n_clusters=10)
    clusterer.train() 