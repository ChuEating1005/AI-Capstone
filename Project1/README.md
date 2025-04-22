# Project1: Clothing Classification Project

This project implements three different machine learning methods for clothing classification:
1. ResNet (Deep Learning-based supervised learning)
2. SVM (Traditional supervised learning)
3. K-means (Unsupervised learning)

## Dataset

The dataset contains 10 types of clothing items, with 50 images per category:
- bag
- dress
- glasses
- hat
- hoodie
- jacket
- pants
- shoes
- sock
- t-shirt

## Project Structure

```
Project1/
├── images/                  # Original dataset
├── dataset/                 # Processed dataset (created by prepare_dataset.py)
│   ├── train/               # 80% of data
│   └── test/                # 20% of data
├── models/                  # Model implementations
│   ├── ResNet.py            # ResNet implementation
│   ├── SVM.py               # SVM implementation
│   └── K_means.py           # K-means implementation
├── plots/                   # Generated visualizations
│   ├── resnet_aug/          # ResNet results and plots with augmentation
│   ├── resnet_noaug/        # ResNet results and plots without augmentation
│   ├── svm_aug/             # SVM results with augmentation
│   ├── svm_noaug/           # SVM results without augmentation
│   └── kmeans/              # K-means clustering results
├── dataset.py               # Dataset classes for loading and augmenting data
├── prepare_dataset.py       # Script to split data into train/test
├── main.py                  # Main script to run all models
└── requirements.txt         # Dependencies
```

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Prepare the dataset:
```
python main.py --prepare
```

## Running the Models

### Run all models:
```
python main.py --all
```

### Run specific models:
```
python main.py --resnet      # Run only ResNet
python main.py --svm         # Run only SVM
python main.py --kmeans      # Run only K-means
```

## Results

The models will generate various output files:
- Model weights: `resnet_best_model.pth`, `svm_model.pkl`, `kmeans_model.pkl`
- Visualizations: Confusion matrices, training curves, cluster visualizations
