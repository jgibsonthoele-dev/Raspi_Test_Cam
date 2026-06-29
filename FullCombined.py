import cv2
import json
import time
import math
from picamera2 import Picamera2

# 1. Initialize Picamera2
picam = Picamera2()

# Configure the camera preview configuration
# For the AI Camera, post-processing models run alongside the camera configuration preview
# Ensure you point to your local imx500 json post-processing configuration file
configure_options = {
    "controls": {
        "NoiseReductionMode": 1
    }
}

# The AI Camera uses a JSON file to map hardware post-processing outputs (like bounding boxes)
# Replace this path with your local imx500 custom network deployment configuration
post_process_config = "/usr/share/rpicam-apps/post_processing/imx500_mobilenet_ssd.json"

picam.configure("preview", configure_options)
picam.start_post_processing(post_process_config)
picam.start()

print("Raspberry Pi AI Camera Pipeline Initialized...")

try:
    while True:
        # Capture the current frame data along with its hardware metadata array
        frame_data = picam.capture_array_with_metadata()
        if frame_data is None:
            continue
            
        frame, metadata = frame_data
        h, w, _ = frame.shape
        
        # Draw frame information
        cv2.putText(frame, "Pi AI Camera Live Stream", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        phones = []
        wrists = []
        
        # 2. Extract Detections directly from Hardware Metadata Tensors
        # The IMX500 deposits post-processing results directly inside the metadata dictionary
        if "post_processing_metadata" in metadata:
            results = metadata["post_processing_metadata"]
            
            # The structure depends on the network configuration file loaded
            # Typically returns a list of dictionaries with 'category', 'conf', and 'box'
            for detection in results.get("detections", []):
                # Class 0 tracking
                category = int(detection.get("category", -1))
                conf = float(detection.get("conf", 0.0))
                
                if conf > 0.30:
                    # Normalized coordinates returned by the sensor: [x_min, y_min, x_max, y_max]
                    box = detection.get("box", [0, 0, 0, 0])
                    x1, y1 = int(box[0] * w), int(box[1] * h)
                    x2, y2 = int(box[2] * w), int(box[3] * h)
                    
                    if category == 0:
                        phones.append((x1, y1, x2, y2))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Phone: {conf:.2f}", (x1, max(20, y1 - 5)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    
                    # If using a multi-class model that includes pose/person landmarks 
                    # extract tracking coordinates here
                    elif category == 1: # Assuming class 1 is person/wrist for the model
                        wrists.append(((x1 + x2) // 2, (y1 + y2) // 2))
                        cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 6, (255, 255, 0), -1)

        # 3. Geometric Proximity Math
        phone_in_hand = False
        for wx, wy in wrists:
            for x1, y1, x2, y2 in phones:
                # Collision box checks prevent resolution-dependent scaling errors
                if (x1 - 20) <= wx <= (x2 + 20) and (y1 - 20) <= wy <= (y2 + 20):
                    phone_in_hand = True
                    break
            if phone_in_hand:
                break

        if phone_in_hand:
            cv2.rectangle(frame, (10, 40), (280, 90), (0, 0, 255), -1)
            cv2.putText(frame, "PHONE IN HAND", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Render the display buffer window
        cv2.imshow("Pi AI Camera Processing", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Safely close hardware hooks on termination
    picam.stop()
    picam.close()
    cv2.destroyAllWindows()