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
from skimage.feature import local_binary_pattern
from scipy.ndimage import sobel
from skimage.feature import hog
import cv2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
import umap

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
        Improved feature extraction with focus on clothing-specific characteristics
        """
        try:
            # Load and resize image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((128, 128))  # 增加圖片尺寸以保留更多細節
            img_array = np.array(img)
            
            # 1. Enhanced Color Features
            # 使用HSV色彩空間
            img_hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # 計算HSV直方圖
            hist_h = np.histogram(img_hsv[:,:,0], bins=16)[0]
            hist_s = np.histogram(img_hsv[:,:,1], bins=16)[0]
            hist_v = np.histogram(img_hsv[:,:,2], bins=16)[0]
            
            # 顏色統計
            color_stats = np.concatenate([
                np.mean(img_hsv, axis=(0,1)),
                np.std(img_hsv, axis=(0,1))
            ])
            
            # 2. 改進的紋理特徵
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # 使用不同尺度的LBP
            lbp_features = []
            for radius in [1, 2, 3]:
                lbp = local_binary_pattern(gray, P=8*radius, R=radius, method='uniform')
                lbp_hist = np.histogram(lbp, bins=10)[0]
                lbp_features.extend(lbp_hist)
            
            # 3. 改進的形狀特徵
            # 使用更密集的HOG特徵
            hog_features = hog(gray, 
                             orientations=9,
                             pixels_per_cell=(16, 16),
                             cells_per_block=(2, 2),
                             visualize=False)
            
            # 4. 邊緣特徵
            edges = cv2.Canny(gray, 100, 200)
            edge_hist = np.histogram(edges, bins=16)[0]
            
            # 組合所有特徵
            features = np.concatenate([
                hist_h/hist_h.sum(), hist_s/hist_s.sum(), hist_v/hist_v.sum(),  # 正規化的顏色直方圖
                color_stats,                                                     # HSV統計
                np.array(lbp_features)/sum(lbp_features),                       # 正規化的LBP特徵
                hog_features/np.linalg.norm(hog_features),                      # 正規化的HOG特徵
                edge_hist/edge_hist.sum()                                       # 正規化的邊緣特徵
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
        
        # Process all data (train, test combined for unsupervised learning)
        for split in ['train', 'test']:
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
        """改進的訓練過程"""
        # 載入數據
        X, y_true, img_paths = self.load_data()
        
        # 特徵選擇
        selector = SelectKBest(score_func=mutual_info_classif, k=100)
        X_selected = selector.fit_transform(X, y_true)
        
        # 標準化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # 使用UMAP進行降維
        self.reducer = umap.UMAP(n_components=30, random_state=42)
        X_reduced = self.reducer.fit_transform(X_scaled)
        
        # 嘗試不同的聚類數量
        silhouette_scores = []
        k_range = range(8, 13)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_reduced)
            score = silhouette_score(X_reduced, labels)
            silhouette_scores.append(score)
        
        # 選擇最佳的聚類數量
        best_k = k_range[np.argmax(silhouette_scores)]
        print(f"Best number of clusters: {best_k}")
        
        # 使用最佳的聚類數量訓練最終模型
        self.model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_labels = self.model.fit_predict(X_reduced)
        
        # 評估和可視化
        self._visualize_clusters(X_reduced, cluster_labels, y_true)
        self._analyze_clusters(cluster_labels, y_true)
        metrics = self.evaluate_clustering(X_reduced, cluster_labels, y_true)
        
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
        plt.savefig('plots/kmeans_clusters_visualization.png')
    
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
        plt.savefig('plots/kmeans_cluster_composition.png')

    def evaluate_clustering(self, X_pca, cluster_labels, y_true):
        """Evaluate the clustering performance"""
        # 1. Calculate ARI
        ari = adjusted_rand_score(y_true, cluster_labels)
        print(f"Adjusted Rand Index: {ari:.4f}")
        
        # 2. Calculate Silhouette Score
        silhouette_avg = silhouette_score(X_pca, cluster_labels)
        print(f"Silhouette Score: {silhouette_avg:.4f}")
        
        # 3. Analyze the purity of each cluster
        purities = []
        for cluster_id in range(self.n_clusters):
            indices = np.where(cluster_labels == cluster_id)[0]
            if len(indices) > 0:
                true_labels = y_true[indices]
                most_common = Counter(true_labels).most_common(1)[0]
                purity = most_common[1] / len(indices)
                purities.append(purity)
        
        avg_purity = np.mean(purities)
        print(f"Average Cluster Purity: {avg_purity:.4f}")
        
        return {
            'ari': ari,
            'silhouette': silhouette_avg,
            'purity': avg_purity
        }

if __name__ == "__main__":
    clusterer = KMeansClusterer(data_dir='dataset', n_clusters=10)
    clusterer.train() 