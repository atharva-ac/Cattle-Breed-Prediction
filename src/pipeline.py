import os
import sys
import json
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detector import CattleDetector
from src.classifier import load_classifier, predict_breed

class CattleInferencePipeline:
    """Integrated 2-Stage Pipeline: Detection + Classification + Knowledge Base Lookup."""
    def __init__(self, weights_path='models/cattle_breed_classifier.pth', breeds_info_path='data/breeds_info.json'):
        self.detector = CattleDetector()
        
        # Load breed knowledge base
        self.breed_database = {}
        if os.path.exists(breeds_info_path):
            with open(breeds_info_path, 'r', encoding='utf-8') as f:
                self.breed_database = json.load(f)

        # Classes supported by trained model
        self.class_names = [
            'Gir', 'Sahiwal', 'Holstein_Friesian', 'Kankrej', 'Ongole', 
            'Hallikar', 'Tharparkar', 'Murrah', 'Jersey', 'Red_Sindhi'
        ]
        
        self.classifier, self.class_names = load_classifier(
            weights_path=weights_path if os.path.exists(weights_path) else None,
            class_names=self.class_names
        )

    def process_image(self, pil_image):
        """
        Executes 2-Stage Pipeline on an input image.
        Returns:
            dict containing detection visualization, crop image, prediction results, and breed profile metadata.
        """
        # Stage 1: Detect cattle and crop region
        annotated_img, cropped_img, bbox, has_detection = self.detector.detect_and_crop(pil_image)

        # Stage 2: Classify cropped cattle region
        prediction = predict_breed(self.classifier, self.class_names, cropped_img)

        # Fetch Breed Knowledge Profile
        predicted_breed_key = prediction["raw_breed"]
        breed_info = self.breed_database.get(predicted_breed_key, {
            "name": predicted_breed_key,
            "native_region": "Indigenous Zebu Region",
            "primary_purpose": "Dairy / Draught",
            "milk_yield": "Indicative: 4-10 litres/day",
            "heat_tolerance": "High climate adaptability",
            "physical_traits": "Distinct Zebu cattle traits",
            "special_characteristics": "Hardy and resilient indigenous breed."
        })

        return {
            "annotated_image": annotated_img,
            "cropped_image": cropped_img,
            "bbox": bbox,
            "has_detection": has_detection,
            "prediction": prediction,
            "breed_info": breed_info
        }
