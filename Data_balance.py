import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ===== Paths =====
train_dir = "training00"
aug_train_dir = "training_augAAAA"

# Check source folder
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Source folder '{train_dir}' not found.")

os.makedirs(aug_train_dir, exist_ok=True)

# ===== Augmentations =====
AUGMENTATIONS = [
   # transforms.RandomRotation(0),  # keep original
   # transforms.RandomRotation(10),  # example augmentation
    transforms.RandomResizedCrop(size=224, scale=(0.8, 0.9)),
]

# Convert each to a Compose
transform_list = [transforms.Compose([aug]) for aug in AUGMENTATIONS]

# ===== Loop through classes =====
for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    aug_class_path = os.path.join(aug_train_dir, class_name)
    os.makedirs(aug_class_path, exist_ok=True)

    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    for img_name in tqdm(images, desc=f"Augmenting {class_name}"):
        img_path = os.path.join(class_path, img_name)
        img = Image.open(img_path).convert("RGB")

        # Apply each transform in list
        for i, transform in enumerate(transform_list):
            aug_img = transform(img)
            aug_name = f"{os.path.splitext(img_name)[0]}_aug{i}.jpg"
            aug_img.save(os.path.join(aug_class_path, aug_name))

print("✅ Augmentation complete. Check folder:", aug_train_dir)
