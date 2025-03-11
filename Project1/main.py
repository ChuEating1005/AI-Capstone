import os
import argparse
import time
from pathlib import Path


def prepare_dataset():
    """Prepare the dataset by splitting it into train, validation, and test sets"""
    from prepare_dataset import main as prepare_main
    prepare_main()

def train_resnet():
    """Train the ResNet model (supervised learning with DL)"""
    from models.ResNet import ResNetTrainer
    
    print("\n" + "="*60)
    print("Training ResNet Model (Supervised Learning - Deep Learning)")
    print("="*60)
    
    start_time = time.time()
    trainer = ResNetTrainer(data_dir='data', batch_size=32, num_epochs=20)
    trainer.train()
    accuracy, _ = trainer.evaluate()
    trainer.plot_training_history()
    end_time = time.time()
    
    print(f"\nResNet training completed in {end_time - start_time:.2f} seconds")
    print(f"Test accuracy: {accuracy:.4f}")
    
    return accuracy

def train_svm():
    """Train the SVM model (supervised learning without DL)"""
    from models.SVM import SVMClassifier
    
    print("\n" + "="*60)
    print("Training SVM Model (Supervised Learning - Traditional ML)")
    print("="*60)
    
    start_time = time.time()
    classifier = SVMClassifier(data_dir='data')
    accuracy, _ = classifier.train()
    end_time = time.time()
    
    print(f"\nSVM training completed in {end_time - start_time:.2f} seconds")
    print(f"Test accuracy: {accuracy:.4f}")
    
    return accuracy

def train_kmeans():
    """Train the K-means model (unsupervised learning)"""
    from models.K_means import KMeansClusterer
    
    print("\n" + "="*60)
    print("Training K-means Model (Unsupervised Learning)")
    print("="*60)
    
    start_time = time.time()
    clusterer = KMeansClusterer(data_dir='data', n_clusters=10)
    clusterer.train()
    end_time = time.time()
    
    print(f"\nK-means training completed in {end_time - start_time:.2f} seconds")
    
    return None  # No standard accuracy for unsupervised learning

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate models for clothing classification')
    parser.add_argument('--prepare', action='store_true', help='Prepare the dataset')
    parser.add_argument('--resnet', action='store_true', help='Train ResNet model')
    parser.add_argument('--svm', action='store_true', help='Train SVM model')
    parser.add_argument('--kmeans', action='store_true', help='Train K-means model')
    parser.add_argument('--all', action='store_true', help='Run all models')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # If no specific arguments are provided, run all
    if not (args.prepare or args.resnet or args.svm or args.kmeans or args.all):
        args.all = True
    
    # Prepare dataset if requested
    if args.prepare or args.all:
        print("\nPreparing dataset...")
        prepare_dataset()
    
    # Check if dataset exists
    if not Path('data').exists():
        print("\nDataset not found. Please run with --prepare first.")
        return
    
    # Ensure models directory exists
    if not Path('models').exists() and (args.resnet or args.svm or args.kmeans or args.all):
        print("\nModels directory not found. Please make sure the 'models' folder exists with model files.")
        return
    
    results = {}
    
    # Train models as requested
    if args.resnet or args.all:
        results['resnet'] = train_resnet()
    
    if args.svm or args.all:
        results['svm'] = train_svm()
    
    if args.kmeans or args.all:
        results['kmeans'] = train_kmeans()
    
    # Print summary of results
    print("\n" + "="*50)
    print("Summary of Results")
    print("="*50)
    
    for model, accuracy in results.items():
        if accuracy is not None:
            print(f"{model.upper()} Test Accuracy: {accuracy:.4f}")
    
    print("\nAll models have been trained and evaluated successfully!")

if __name__ == "__main__":
    main() 