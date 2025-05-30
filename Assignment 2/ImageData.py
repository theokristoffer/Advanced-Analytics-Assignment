from torch.utils.data import Dataset
import os
from PIL import Image


class ImageData(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.mapping = {folder: idx for idx, folder in enumerate(sorted(os.listdir(data_dir)))}

        self.classes = list(self.mapping.keys())

        self.image_paths = []
        self.labels = []

        for folder in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, folder)
            if os.path.isdir(folder_path):
                for img_file in os.listdir(folder_path):
                    if img_file.endswith(".jpg"):
                        self.image_paths.append(os.path.join(folder_path, img_file))
                        self.labels.append(self.mapping[folder])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return img, label
