"""YOLOv8 ONNX detector with CPU fallback behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

import config

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - keeps app usable on systems without ORT.
    ort = None


COCO_NAMES = {
    0: "person",
    67: "cell phone",
}


@dataclass
class Detection:
    label: str
    confidence: float
    box: Tuple[int, int, int, int]


class YoloV8OnnxDetector:
    """Runs YOLOv8 ONNX and returns filtered person/phone detections."""

    def __init__(self, model_path: Path | str = config.MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self.session = None
        self.input_name = ""
        self.input_hw = config.YOLO_INPUT_SIZE
        self.enabled = False

        if ort is None or not self.model_path.exists() or self.model_path.stat().st_size == 0:
            print("ONNX detector disabled: install onnxruntime and place model/phone_detector.onnx to enable detections.")
            return

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        shape = input_meta.shape
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self.input_hw = (shape[3], shape[2])
        self.enabled = True

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if not self.enabled or self.session is None:
            return []

        original_h, original_w = frame_bgr.shape[:2]
        tensor, scale, pad_x, pad_y = self._preprocess(frame_bgr)
        outputs = self.session.run(None, {self.input_name: tensor})
        detections = self._postprocess(outputs[0], original_w, original_h, scale, pad_x, pad_y)
        return detections

    def _preprocess(self, frame_bgr: np.ndarray):
        input_w, input_h = self.input_hw
        h, w = frame_bgr.shape[:2]
        scale = min(input_w / w, input_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        pad_x = (input_w - new_w) // 2
        pad_y = (input_h - new_h) // 2
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        return tensor, scale, pad_x, pad_y

    def _postprocess(self, raw: np.ndarray, original_w: int, original_h: int, scale: float, pad_x: int, pad_y: int) -> List[Detection]:
        pred = np.squeeze(raw)
        if pred.ndim != 2:
            return []

        # YOLOv8 exports commonly return [84, 8400]; transpose to [8400, 84].
        if pred.shape[0] < pred.shape[1] and pred.shape[0] in (6, 84, 85):
            pred = pred.T

        boxes: List[Tuple[int, int, int, int]] = []
        scores: List[float] = []
        labels: List[str] = []

        for row in pred:
            parsed = self._parse_row(row)
            if parsed is None:
                continue
            cx, cy, bw, bh, conf, class_id = parsed
            label = COCO_NAMES.get(class_id, "phone" if class_id == 1 else str(class_id))
            if label not in config.DETECTION_CLASSES:
                continue

            x1 = int((cx - bw / 2 - pad_x) / scale)
            y1 = int((cy - bh / 2 - pad_y) / scale)
            x2 = int((cx + bw / 2 - pad_x) / scale)
            y2 = int((cy + bh / 2 - pad_y) / scale)
            x1 = max(0, min(original_w - 1, x1))
            y1 = max(0, min(original_h - 1, y1))
            x2 = max(0, min(original_w - 1, x2))
            y2 = max(0, min(original_h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((x1, y1, x2, y2))
            scores.append(float(conf))
            labels.append("phone" if label == "cell phone" else label)

        keep = self._nms(boxes, scores, config.DETECTION_IOU_THRESHOLD)
        return [Detection(labels[i], scores[i], boxes[i]) for i in keep]

    def _parse_row(self, row: Sequence[float]):
        if len(row) >= 6 and len(row) <= 7:
            conf = float(row[4])
            class_id = int(row[5])
            if conf < config.DETECTION_CONFIDENCE:
                return None
            return float(row[0]), float(row[1]), float(row[2]), float(row[3]), conf, class_id

        class_scores = np.asarray(row[4:], dtype=np.float32)
        class_id = int(np.argmax(class_scores))
        conf = float(class_scores[class_id])
        if conf < config.DETECTION_CONFIDENCE:
            return None
        return float(row[0]), float(row[1]), float(row[2]), float(row[3]), conf, class_id

    @staticmethod
    def _nms(boxes: List[Tuple[int, int, int, int]], scores: List[float], iou_threshold: float) -> List[int]:
        if not boxes:
            return []
        b = np.array(boxes, dtype=np.float32)
        s = np.array(scores, dtype=np.float32)
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = s.argsort()[::-1]
        keep: List[int] = []

        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[np.where(iou <= iou_threshold)[0] + 1]
        return keep
