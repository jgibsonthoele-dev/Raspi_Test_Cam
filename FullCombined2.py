import cv2
import time
import math
from picamera2 import Picamera2
# FIX: The correct location for the AI Camera hardware module
from picamera2.devices import IMX500

# 1. Load the IMX500 model handler 
# Point to your packaged network file (.rpk). The default MobileNet-SSD is used here.
model_path = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
imx500 = IMX500(model_path)

# 2. Initialize Picamera2 using the specific AI camera hardware ID
picam = Picamera2(imx500.camera_num)

# 3. Create and apply the camera configuration
config = picam.create_preview_configuration()
picam.configure(config)

# Start the hardware pipeline
picam.start()
print("Raspberry Pi AI Camera pipeline initialized successfully...")

try:
    while True:
        # Capture the raw camera matrix array along with its synced tensor metadata
        frame = picam.capture_array()
        metadata = picam.capture_metadata()
        
        if frame is None or metadata is None:
            continue
            
        h, w, _ = frame.shape
        phones = []
        wrists = []

        # 4. Extract outputs directly from the IMX500 hardware metadata
        # imx500.get_outputs parses the raw tensor stream natively returned by the camera
        outputs = imx500.get_outputs(metadata)
        
        if outputs is not None:
            # Note: The parsing structure depends completely on your compiled .rpk network layout.
            # For standard Object Detection models, it outputs bounding boxes and classes.
            for detection in outputs:
                # Extract your tracking classes (assuming 0 is your target category)
                category = int(detection.get("category", -1))
                conf = float(detection.get("conf", 0.0))
                
                if conf > 0.30:  # Confidence threshold filter
                    # Convert normalized camera coordinates to pixel coordinates
                    coords = detection.get("box", [0, 0, 0, 0])
                    x1, y1 = int(coords[0] * w), int(coords[1] * h)
                    x2, y2 = int(coords[2] * w), int(coords[3] * h)
                    
                    if category == 0:  # Matches your custom model's phone ID
                        phones.append((x1, y1, x2, y2))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Phone: {conf:.2f}", (x1, max(20, y1 - 5)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    elif category == 1:  # Assuming Class 1 maps to people/wrists
                        wx, wy = (x1 + x2) // 2, (y1 + y2) // 2
                        wrists.append((wx, wy))
                        cv2.circle(frame, (wx, wy), 6, (255, 255, 0), -1)

        # 5. Proximity Match (Phone in Hand)
        phone_in_hand = False
        for wx, wy in wrists:
            for x1, y1, x2, y2 in phones:
                # Check if the calculated center point is inside the phone bounding box
                if (x1 - 20) <= wx <= (x2 + 20) and (y1 - 20) <= wy <= (y2 + 20):
                    phone_in_hand = True
                    break
            if phone_in_hand:
                break

        if phone_in_hand:
            cv2.rectangle(frame, (10, 10), (280, 60), (0, 0, 255), -1)
            cv2.putText(frame, "PHONE IN HAND", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display Live Canvas Output
        cv2.imshow("Pi AI Camera Inference Window", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Safely close hardware links on termination
    picam.stop()
    picam.close()
    cv2.destroyAllWindows()