import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DentalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 각 질병 (category_id_3) 폴더에 있는 이미지를 모두 리스트에 저장
        for label_folder in os.listdir(root_dir):
            label_folder_path = os.path.join(root_dir, label_folder)
            if os.path.isdir(label_folder_path):
                # 폴더 이름을 라벨로 사용 (category_id_3)
                label = int(label_folder)  
                # 해당 폴더에 있는 모든 이미지 경로를 저장
                for image_file in os.listdir(label_folder_path):
                    if image_file.endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(label_folder_path, image_file)
                        self.image_paths.append(image_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),          
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])


def get_data_loaders(train_dir, val_dir, test_dir, batch_size=32, num_workers=4):
    transform = get_transforms()

    train_dataset = DentalDataset(train_dir, transform=transform)
    val_dataset = DentalDataset(val_dir, transform=transform)
    test_dataset = DentalDataset(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def main():
    
    root_dir = "cropped_images"
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")
    test_dir = os.path.join(root_dir, "test")

    train_loader, val_loader, test_loader = get_data_loaders(train_dir, val_dir, test_dir)

    for images, labels in train_loader:
        print(f"Images batch shape: {images.shape}")
        print(f"Labels: {labels}")
        break 

if __name__ == "__main__":
    main()
