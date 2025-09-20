import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
dataset_dir = "Faulty_solar_panel_dataset"
train_dir = "training"
test_dir = "testing"

# Create output dirs
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Split ratio
split_ratio = 0.8  # 80% train, 20% test

# Loop through each class
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # List images
    images = os.listdir(class_path)

    # Split into train/test (different split each run)
    train_files, test_files = train_test_split(images, train_size=split_ratio)

    # Make subfolders in train/test
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # Copy train images
    for file_name in train_files:
        shutil.copy2(os.path.join(class_path, file_name), os.path.join(train_dir, class_name, file_name))

    # Copy test images
    for file_name in test_files:
        shutil.copy2(os.path.join(class_path, file_name), os.path.join(test_dir, class_name, file_name))

print("✅ Dataset split completed: 80% training / 20% testing")
