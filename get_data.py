import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

source_dataset = "coindataset"
base_dest_dir = "dataset"

train_dir = os.path.join(base_dest_dir, "train")
val_dir = os.path.join(base_dest_dir, "val")
test_dir = os.path.join(base_dest_dir, "test")

for split_dir in [train_dir, val_dir, test_dir]:
    for label in os.listdir(source_dataset):
        os.makedirs(os.path.join(split_dir, label), exist_ok=True)

image_paths = []
labels = []

for label in os.listdir(source_dataset):
    label_path = os.path.join(source_dataset, label)
    if os.path.isdir(label_path):
        for image in os.listdir(label_path):
            image_paths.append(os.path.join(label_path, image))
            labels.append(label)

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

def move_files(file_paths, file_labels, dest_dir):
    for path, label in zip(file_paths, file_labels):
        dest_path = os.path.join(dest_dir, label, os.path.basename(path))
        shutil.copy2(path, dest_path)

move_files(train_paths, train_labels, train_dir)
move_files(val_paths, val_labels, val_dir)
move_files(test_paths, test_labels, test_dir)

print(f"Train set: {len(train_paths)} images")
print(f"Validation set: {len(val_paths)} images")
print(f"Test set: {len(test_paths)} images")

print(f"Dataset organized successfully in '{base_dest_dir}'!")
