import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# Default confidence threshold below which prediction is marked uncertain
CONFIDENCE_THRESHOLD = 0.45

class CattleBreedClassifier(nn.Module):
    """Deep learning transfer-learning classification model for cattle breeds."""
    def __init__(self, num_classes=10, backbone='mobilenet_v3'):
        super(CattleBreedClassifier, self).__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        if backbone == 'mobilenet_v3':
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            self.model = models.mobilenet_v3_small(weights=weights)
            # Freeze feature backbone
            for param in self.model.features.parameters():
                param.requires_grad = False

            in_features = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features, num_classes)
            )
        elif backbone == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT
            self.model = models.resnet18(weights=weights)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features, num_classes)
            )
        else:
            # Fallback to EfficientNet-B0
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.model = models.efficientnet_b0(weights=weights)
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features, num_classes)
            )

    def forward(self, x):
        return self.model(x)

def load_classifier(weights_path=None, class_names=None, device='cpu'):
    """Instantiates and loads the classification model."""
    if class_names is None:
        class_names = [
            'Gir', 'Sahiwal', 'Holstein_Friesian', 'Kankrej', 'Ongole', 
            'Hallikar', 'Tharparkar', 'Murrah', 'Jersey', 'Red_Sindhi'
        ]
    
    model = CattleBreedClassifier(num_classes=len(class_names))
    model.to(device)
    
    if weights_path and os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"Loaded classifier weights from {weights_path}")
        except Exception as e:
            print(f"Warning: Could not load weights from {weights_path}: {e}")
            model.eval()
    else:
        model.eval()

    return model, class_names

def predict_breed(model, class_names, pil_image, device='cpu', confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    Runs classification inference on a cropped PIL Image.
    Returns predicted breed name, confidence percentage (0-100), status, and top class probabilities.
    """
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = preprocess(pil_image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    top_prob, top_idx = torch.topk(probabilities, k=min(5, len(class_names)))
    
    top_prob = top_prob.cpu().numpy()
    top_idx = top_idx.cpu().numpy()

    best_class = class_names[top_idx[0]]
    best_conf = float(top_prob[0])

    top_predictions = {class_names[idx]: float(prob) for idx, prob in zip(top_idx, top_prob)}

    if best_conf < confidence_threshold:
        status = "Low Confidence Prediction (Uncertain)"
        display_breed = "Unknown / Other Breed"
    else:
        status = "High-confidence prediction"
        display_breed = best_class

    return {
        "predicted_breed": display_breed,
        "raw_breed": best_class,
        "confidence": round(best_conf * 100, 1),
        "status": status,
        "top_predictions": top_predictions
    }
