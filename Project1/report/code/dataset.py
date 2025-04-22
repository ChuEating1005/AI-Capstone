from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from skimage.feature import hog
import umap

data_aug = True

train_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]) if data_aug else transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

svm_tfm = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomGrayscale(p=0.1),
])


class ResnetDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.classes = []
        self.loader = Image.open

        # Populate samples and classes
        for class_name in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                self.classes.append(class_name)
                for file_name in os.listdir(class_path):
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(class_path, file_name)
                        self.samples.append((file_path, self.classes.index(class_name)))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        Override the __getitem__ method to skip corrupted images
        """
        path, label = self.samples[index]
        try:
            sample = self.loader(path)

            if self.transform is not None:
                sample = self.transform(sample)

            return sample, label
        
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Return a random valid image instead
            valid_idx = (index + 1) % len(self)
            return self.__getitem__(valid_idx)

class SVMDataset:
    def __init__(self, data_dir, augmentations=3):
        self.data_dir = data_dir
        self.augmentations = augmentations
        self.train_transform = svm_tfm
        self.test_transform = test_tfm
        self.class_names = []
        self.class_to_idx = {}

    def load_data(self):
        X_train, y_train = [], []
        X_test, y_test = [], []

        # Get class names from train directory
        self.class_names = [d for d in os.listdir(os.path.join(self.data_dir, 'train')) 
                            if os.path.isdir(os.path.join(self.data_dir, 'train', d))]
        
        # Create class to index mapping
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        
        # Process training data with augmentation
        print("Loading training data with augmentation...")
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, 'train', class_name)
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    img = Image.open(img_path).convert('RGB')
                    
                    # Original image
                    features = self.extract_features(img, augment=False)
                    if features is not None:
                        X_train.append(features)
                        y_train.append(self.class_to_idx[class_name])
                    
                    # Augmented versions
                    for _ in range(self.augmentations):
                        aug_features = self.extract_features(img, augment=True)
                        if aug_features is not None:
                            X_train.append(aug_features)
                            y_train.append(self.class_to_idx[class_name])
        
        # Process test data (no augmentation)
        print("Loading test data...")
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, 'test', class_name)
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    img = Image.open(img_path).convert('RGB')
                    features = self.extract_features(img, augment=False)
                    if features is not None:
                        X_test.append(features)
                        y_test.append(self.class_to_idx[class_name])
        
        return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

    def extract_features(self, img, augment=False):
        if augment:
            img = self.train_transform(img)

        # Resize to 128x128
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
        
        # Convert to 8-bit unsigned integer
        gray = (gray * 255).astype(np.uint8)

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
        
        # Combine all features and normalize
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
