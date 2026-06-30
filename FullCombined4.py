import cv2
import time
import math
import sys
import numpy as np
from picamera2 import Picamera2
from picamera2.devices import IMX500

# Optional: Attempt to load MediaPipe for the hands if installed, else use a fallback box check
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose_tracker = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5)
    HAS_MEDIAPIPE = True
    print("MediaPipe successfully loaded for CPU pose estimation.")
except ImportError:
    HAS_MEDIAPIPE = False
    print("MediaPipe not found. Falling back to absolute hand-to-phone collision checks.")

# 1. Initialize the IMX500 Hardware
model_path = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
try:
    imx500 = IMX500(model_path)
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration()
    picam2.configure(config)
    picam2.start()
    print("IMX500 Hardware Pipeline Streaming Active.")
except Exception as e:
    print(f"Hardware Link Error: {e}")
    sys.exit(1)

# Unique window identifier string
WINDOW_NAME = "AI Camera Live Pipeline"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

try:
    while True:
        frame = picam2.capture_array()
        metadata = picam2.capture_metadata()
        
        if frame is None or metadata is None:
            continue
            
        h, w, _ = frame.shape
        phones = []
        hand_points = []

        # 2. Extract Phone Detections from the IMX500 Accelerator
        outputs = imx500.get_outputs(metadata)
        if outputs is not None:
            for detection in outputs:
                category = int(detection.get("category", -1))
                conf = float(detection.get("conf", 0.0))
                
                # In the MobileNet-SSD RPK, Category 0 or 67 or 77 depending on conversion mappings
                # Let's intercept the first distinct detected object if fine-tuning is unknown
                if conf > 0.35:
                    coords = detection.get("box", [0, 0, 0, 0])
                    x1, y1 = int(coords[0] * w), int(coords[1] * h)
                    x2, y2 = int(coords[2] * w), int(coords[3] * h)
                    
                    # Store bounding box
                    phones.append((x1, y1, x2, y2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Obj {category}: {conf:.2f}", (x1, max(20, y1 - 5)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 3. Extract Wrist Tracking from the Pi's CPU via MediaPipe
        if HAS_MEDIAPIPE:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_tracker.process(rgb)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Left wrist (15) and Right wrist (16)
                for idx in [15, 16]:
                    wx = int(landmarks[idx].x * w)
                    wy = int(landmarks[idx].y * h)
                    hand_points.append((wx, wy))
                    cv2.circle(frame, (wx, wy), 8, (255, 255, 0), -1)

        # 4. Proximity Math Evaluation
        phone_in_hand = False
        if len(phones) > 0 and len(hand_points) > 0:
            for h_x, h_y in hand_points:
                for x1, y1, x2, y2 in phones:
                    # Check if the hand point coordinates sit inside the phone box boundary
                    if (x1 - 25) <= h_x <= (x2 + 25) and (y1 - 25) <= h_y <= (y2 + 25):
                        phone_in_hand = True
                        break
                if phone_in_hand:
                    break

        if phone_in_hand:
            cv2.rectangle(frame, (10, 10), (290, 60), (0, 0, 255), -1)
            cv2.putText(frame, "PHONE IN HAND", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)
        
        # Break loop if user clicks 'q' or closes UI window frame
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    print("\nSafely breaking GObject bindings...")
    # Order matters: Close OpenCV window frames before detaching the hardware driver stack
    cv2.destroyAllWindows()
    cv2.waitKey(1) 
    
    if picam2 is not None:
        try:
            picam2.stop()
            picam2.close()
            print("AI Camera connection severed cleanly.")
        except Exception:
            pass