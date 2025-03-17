from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

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
