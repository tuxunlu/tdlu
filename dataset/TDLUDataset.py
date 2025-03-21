import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class TDLUDataset(Dataset):
    def __init__(self, image_dir, csv_path, transform=None):
        self.image_dir = image_dir
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

        # Load the CSV file
        self.csv = pd.read_csv(csv_path)
        self.csv["subject_id"] = self.csv["subject_id"].astype(str)

        # Get the list of image files
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]

        # For each image in the self.image_files, create a entry dictionary mapping subject_id to tdlu_count_final
        self.label_map = {}
        for img_file in self.image_files:
            subject_id = img_file.split("-")[0]  # Remove file extension
            # Convert the subject_id to string (if not already)
            subject_id = str(subject_id)
            
            # Find the row in the CSV with the matching subject_id.
            row = self.csv[self.csv["subject_id"] == subject_id]
            if not row.empty:
                # Get the corresponding tdlu_count_final value.
                tdlu_value = row.iloc[0]["tdlu_count_final"]
                self.label_map[subject_id] = tdlu_value
            else:
                # Optional: handle missing CSV entries for an image
                print(f"Warning: subject_id {subject_id} not found in CSV.")


    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        subject_id = image_path.split("-")[0]

        image_path = os.path.join(self.image_dir, image_path)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if subject_id not in self.label_map:
            raise ValueError(f"Subject ID {subject_id} not found in the label map.")
        label = int(self.label_map[subject_id])

        return image, label