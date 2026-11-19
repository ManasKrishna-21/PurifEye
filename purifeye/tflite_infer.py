"""AutoML MobileNet-SSD TFLite inference engine with multithreading and tile batching."""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType


class TFLiteDetector:
    """Wrapper for the 2.76 MB AutoML MobileNet-SSD TFLite model."""

    def __init__(self, model_path: str | Path, num_threads: int = 4):
        self.model_path = str(model_path)
        self.num_threads = num_threads
        # Use BUILTIN_WITHOUT_DEFAULT_DELEGATES to prevent incompatible XNNPack on older models
        self.interpreter = Interpreter(
            model_path=self.model_path,
            experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
            num_threads=self.num_threads
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]["shape"]  # [1, 512, 512, 3]
        self.input_h = int(self.input_shape[1])
        self.input_w = int(self.input_shape[2])
        self.input_idx = self.input_details[0]["index"]

        # Output indices: boxes (1, 57, 4), classes (1, 57), scores (1, 57), num (1,)
        self.out_boxes_idx = self.output_details[0]["index"]
        self.out_classes_idx = self.output_details[1]["index"]
        self.out_scores_idx = self.output_details[2]["index"]
        self.out_num_idx = self.output_details[3]["index"]

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.07
    ) -> List[Dict[str, Any]]:
        """Run single image inference.

        image: BGR or RGB image numpy array.
        Returns: list of dicts with 'box' [x1, y1, x2, y2] (pixel coordinates), 'confidence', 'class_id'.
        """
        orig_h, orig_w = image.shape[:2]
        resized = cv2.resize(image, (self.input_w, self.input_h))
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        elif resized.shape[2] == 4:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
        else:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
        self.interpreter.set_tensor(self.input_idx, input_data)
        self.interpreter.invoke()

        raw_boxes = self.interpreter.get_tensor(self.out_boxes_idx)[0]      # [57, 4]: ymin, xmin, ymax, xmax (normalized)
        raw_classes = self.interpreter.get_tensor(self.out_classes_idx)[0]  # [57]
        raw_scores = self.interpreter.get_tensor(self.out_scores_idx)[0]    # [57]
        num_det = int(self.interpreter.get_tensor(self.out_num_idx)[0])

        detections = []
        for i in range(min(num_det, len(raw_scores))):
            score = float(raw_scores[i])
            if score < conf_threshold:
                continue
            ymin, xmin, ymax, xmax = raw_boxes[i]
            # Convert normalized to pixel coordinates of original image
            x1 = max(0.0, float(xmin * orig_w))
            y1 = max(0.0, float(ymin * orig_h))
            x2 = min(float(orig_w), float(xmax * orig_w))
            y2 = min(float(orig_h), float(ymax * orig_h))

            detections.append({
                "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "confidence": round(score, 4),
                "class_id": int(raw_classes[i]),
            })

        return detections
