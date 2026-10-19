import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

class CattleDetector:
    """Stage 1 YOLO Object Detection model for localizing cattle in input images."""
    def __init__(self, model_name='yolov8n.pt'):
        try:
            self.model = YOLO(model_name)
            # Class 19 in COCO model is 'cow' / 'cattle', Class 16 is 'dog', 17 'horse', 18 'sheep', 19 'cow', 20 'elephant', 21 'bear'
            self.target_classes = [19, 17, 18, 20] # cow, horse, sheep, elephant
        except Exception as e:
            print(f"Error loading YOLO model ({model_name}): {e}")
            self.model = None

    def detect_and_crop(self, pil_image):
        """
        Detects cattle in a PIL Image.
        Returns:
            annotated_image: PIL Image with bounding box drawn
            cropped_cattle: PIL Image cropped to detected cattle region
            box_coords: tuple (xmin, ymin, xmax, ymax)
            has_detection: bool
        """
        img_np = np.array(pil_image)
        height, width, _ = img_np.shape

        if self.model is None:
            # Fallback if model failed to load
            return pil_image, pil_image, (0, 0, width, height), False

        # Convert RGB to BGR for OpenCV / YOLO processing
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        results = self.model(img_bgr, verbose=False)
        boxes = results[0].boxes

        best_box = None
        best_conf = 0.0

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                # Prioritize cattle/cow class
                if cls_id in self.target_classes or cls_id == 19:
                    if conf > best_conf:
                        best_conf = conf
                        best_box = box.xyxy[0].cpu().numpy()

            # If no cattle class matched, pick the largest detection box available
            if best_box is None:
                max_area = 0
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                    if area > max_area:
                        max_area = area
                        best_box = xyxy
                        best_conf = float(box.conf[0].item())

        annotated_bgr = img_bgr.copy()
        
        if best_box is not None:
            xmin, ymin, xmax, ymax = map(int, best_box)

            # Pad bounding box slightly for context
            pad_x = int((xmax - xmin) * 0.05)
            pad_y = int((ymax - ymin) * 0.05)

            xmin_crop = max(0, xmin - pad_x)
            ymin_crop = max(0, ymin - pad_y)
            xmax_crop = min(width, xmax + pad_x)
            ymax_crop = min(height, ymax + pad_y)

            cropped_np = img_np[ymin_crop:ymax_crop, xmin_crop:xmax_crop]
            cropped_pil = Image.fromarray(cropped_np)

            # Draw bounding box and label on original image
            cv2.rectangle(annotated_bgr, (xmin, ymin), (xmax, ymax), (0, 215, 255), 3)
            label = f"Cattle Detected ({best_conf*100:.1f}%)"
            
            # Label background box
            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_bgr, (xmin, ymin - h_text - 10), (xmin + w_text + 10, ymin), (0, 215, 255), -1)
            cv2.putText(annotated_bgr, label, (xmin + 5, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            annotated_pil = Image.fromarray(annotated_rgb)

            return annotated_pil, cropped_pil, (xmin, ymin, xmax, ymax), True
        else:
            # No detection: return original image as crop
            return pil_image, pil_image, (0, 0, width, height), False
