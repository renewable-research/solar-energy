import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import time
1
1
# ================== Paths ==================
img_dir = "img"   # folder containing images for testing
ckpt_path = "checkpoints/xception_state_dict.ckpt"  # trained model weights
train_dir = "training_aug"  # folder with class subfolders used during training

# ================== Device ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== Transforms ==================
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ================== Load Model ==================
num_classes = len(os.listdir(train_dir))  # assumes training_aug/<class_folders>
import torch
import torch.nn as nn
import timm

# Load pretrained Xception
model = timm.create_model("xception", pretrained=True)

# Get number of features of last layer
num_ftrs = model.get_classifier().in_features

# Replace classifier with your custom head
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)


# Load trained weights
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model = model.to(device)
model.eval()

# Get class names from training folder
class_names = sorted(os.listdir(train_dir))

# ================== Inference ==================
print("\n🔎 Running inference on folder:", img_dir)
times = []
count = 0

for img_file in os.listdir(img_dir):
    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(img_dir, img_file)
        
        # Load & preprocess
        image = Image.open(img_path).convert("RGB")
        input_tensor = data_transform(image).unsqueeze(0).to(device)

        # Inference
        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_tensor)
            preds = outputs.argmax(dim=1).item()
        end_time = time.time()

        elapsed_ms = (end_time - start_time) * 1000
        if count > 10:
            times.append(elapsed_ms)
       
        count += 1

        pred_class = class_names[preds]
        print(f"🖼️ {img_file} --> {pred_class}  (time: {elapsed_ms:.2f} ms)")

if count > 10:
    avg_time = sum(times) / len(times)
    print(f"\n⏱️ Average inference time per image: {avg_time:.2f} ms")
else:
    print("\n⚠️ No valid images found in folder.")
def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

# Backbone = everything except the classifier
backbone_params = count_params(model) - count_params(model.fc)

# Classifier = the fc head
classifier_params = count_params(model.fc)

# Total
total_params = backbone_params + classifier_params

print("\n🔹 Xception Parameters:")
print(f"Backbone params   : {backbone_params/1e6:.2f} M")
print(f"Classifier params : {classifier_params/1e6:.2f} M")
print(f"Total params      : {total_params/1e6:.2f} M")

