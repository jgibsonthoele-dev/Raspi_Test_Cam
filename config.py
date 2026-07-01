"""Runtime configuration for the driver attentiveness monitor."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "phone_detector.onnx"

# Camera / processing dimensions. 640x480 keeps Raspberry Pi 5 CPU load practical.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 30
DISPLAY_WINDOW = "Driver attentiveness monitor"

# YOLOv8 ONNX input. Change only if your exported model uses a different size.
YOLO_INPUT_SIZE = (640, 640)
DETECTION_CONFIDENCE = 0.35
DETECTION_IOU_THRESHOLD = 0.45
DETECTION_CLASSES = {"person", "phone", "cell phone"}

# Phone-in-hand logic.
PHONE_WRIST_DISTANCE_PX = 120
PHONE_CONFIRM_SECONDS = 0.5

# Attentiveness scoring thresholds / penalties.
PHONE_PENALTY_TABLE = (
    (1.0, 0),
    (3.0, 10),
    (5.0, 25),
    (float("inf"), 40),
)
EYE_AWAY_PENALTY_TABLE = (
    (2.0, 0),
    (4.0, 15),
    (float("inf"), 30),
)
HEAD_AWAY_SECONDS = 2.0
HEAD_AWAY_PENALTY = 20

# Face orientation heuristics. These are intentionally conservative because
# Face Mesh is not a calibrated 3-D head-pose solver.
EYE_AWAY_RATIO_THRESHOLD = 0.19
HEAD_AWAY_X_THRESHOLD = 0.18
HEAD_AWAY_Y_THRESHOLD = 0.22

# MediaPipe confidence thresholds.
POSE_DETECTION_CONFIDENCE = 0.55
POSE_TRACKING_CONFIDENCE = 0.55
FACE_DETECTION_CONFIDENCE = 0.55
FACE_TRACKING_CONFIDENCE = 0.55

# Optional GPIO buzzer.
ENABLE_BUZZER = False
BUZZER_GPIO_PIN = 18
BUZZER_SCORE_THRESHOLD = 60
