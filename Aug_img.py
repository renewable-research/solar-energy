import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import imgaug.augmenters as iaa
 

# ===== Paths =====
train_dir = "training"
aug_train_dir = "training_aug"
os.makedirs(aug_train_dir, exist_ok=True)

# ===== Augmentations =====
AUGMENTATIONS = [
    transforms.RandomRotation(0),  # include original
    transforms.RandomHorizontalFlip(p=1.0),
   
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
   
    transforms.GaussianBlur(kernel_size=5),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 0.9)),
]

 

# Convert to torchvision Compose for easier use
transform_list = [transforms.Compose([aug]) for aug in AUGMENTATIONS]

# ===== Loop through classes =====
for class_name in os.listdir(train_dir):
    class_path = os.path.join(train_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # Make class folder in augmented training folder
    aug_class_path = os.path.join(aug_train_dir, class_name)
    os.makedirs(aug_class_path, exist_ok=True)

    images = os.listdir(class_path)
    for img_name in tqdm(images, desc=f"Augmenting {class_name}"):
        img_path = os.path.join(class_path, img_name)
        img = Image.open(img_path).convert("RGB")

        # Apply each augmentation
        for i, transform in enumerate(transform_list):
            aug_img = transform(img)
            aug_name = f"{os.path.splitext(img_name)[0]}_aug{i}.jpg"
            aug_img.save(os.path.join(aug_class_path, aug_name))
