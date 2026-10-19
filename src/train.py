import os
import sys
import random

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.dataset import CattleBreedDataset, get_transforms
from src.classifier import CattleBreedClassifier

TARGET_BREEDS = [
    'Gir', 'Sahiwal', 'Holstein_Friesian', 'Kankrej', 'Ongole', 
    'Hallikar', 'Tharparkar', 'Murrah', 'Jersey', 'Red_Sindhi'
]

def collect_image_paths(base_dir='.', target_breeds=TARGET_BREEDS, max_samples_per_breed=60):
    """Gathers image filepaths and corresponding class labels for target breeds."""
    image_paths = []
    labels = []
    class_to_idx = {breed: idx for idx, breed in enumerate(target_breeds)}

    print("Collecting dataset samples for target breeds...")
    for breed in target_breeds:
        breed_dir = os.path.join(base_dir, breed)
        if not os.path.isdir(breed_dir):
            print(f"Warning: Directory {breed_dir} not found. Skipping.")
            continue

        files = [
            os.path.join(breed_dir, f) for f in os.listdir(breed_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
        
        # Shuffle and sample up to max_samples_per_breed for fast high-quality training
        random.seed(42)
        random.shuffle(files)
        sampled_files = files[:max_samples_per_breed]

        for path in sampled_files:
            image_paths.append(path)
            labels.append(class_to_idx[breed])

        print(f"  - {breed}: loaded {len(sampled_files)} images (Class index: {class_to_idx[breed]})")

    return image_paths, labels, target_breeds

def train_model(epochs=8, batch_size=32, lr=0.003, save_path='models/cattle_breed_classifier.pth'):
    """Trains/fine-tunes the Cattle Breed Classifier and saves model weights."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    image_paths, labels, class_names = collect_image_paths()
    if len(image_paths) == 0:
        print("Error: No images found for training!")
        return

    # Train / Val Split (80% / 20%)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print(f"\nTotal Dataset Size: {len(image_paths)} images")
    print(f"  - Training samples: {len(train_paths)}")
    print(f"  - Validation samples: {len(val_paths)}")

    train_dataset = CattleBreedDataset(train_paths, train_labels, transform=get_transforms(is_train=True))
    val_dataset = CattleBreedDataset(val_paths, val_labels, transform=get_transforms(is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model
    model = CattleBreedClassifier(num_classes=len(class_names), backbone='mobilenet_v3').to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    print("\nStarting Training Loop...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)

        epoch_loss = running_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100.0

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == targets).sum().item()
                total_val += targets.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100.0

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        # Save best model
        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best checkpoint to {save_path} (Val Acc: {best_val_acc:.2f}%)")

    print(f"\nTraining Complete! Best Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == '__main__':
    train_model(epochs=6, batch_size=16)
