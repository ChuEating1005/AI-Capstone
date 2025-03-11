# Clothing Classification Project

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
│   ├── train/               # 70% of data
│   ├── val/                 # 20% of data
│   └── test/                # 10% of data
├── prepare_dataset.py       # Script to split data into train/val/test
├── resnet_model.py          # ResNet implementation (supervised DL)
├── svm_model.py             # SVM implementation (supervised non-DL)
├── kmeans_model.py          # K-means implementation (unsupervised)
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

## Author

I-TING CHU (111550093)
National Yang Ming Chiao Tung University (NYCU) 