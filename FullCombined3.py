import cv2
import time
import math
import sys
from picamera2 import Picamera2
from picamera2.devices import IMX500 

# Initialize variables globally as None to prevent g_object clean up assertions if things fail early
imx500 = None
picam2 = None

try:
    print("Initializing Sony IMX500 Hardware Coprocessor...")
    model_path = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
    
    # Safely wrap hardware mapping
    imx500 = IMX500(model_path)
    
    print(f"Connecting to Camera Device Node #{imx500.camera_num}...")
    picam2 = Picamera2(imx500.camera_num)

    config = picam2.create_preview_configuration()
    picam2.configure(config)
    
    print("Starting Camera Pipeline Stream...")
    picam2.start()
    
except Exception as e:
    print(f"\n[CRITICAL INITIALIZATION ERROR]: {e}")
    print("Aborting to prevent GLib-GObject resource cleanup crashes.")
    # Exit immediately before triggering the final script block unrefs
    sys.exit(1)

print("Pipeline active. Press 'q' in the window to quit.")

try:
    while True:
        frame = picam2.capture_array()
        metadata = picam2.capture_metadata()
        
        if frame is None or metadata is None:
            continue
            
        h, w, _ = frame.shape
        phones = []
        wrists = []

        outputs = imx500.get_outputs(metadata)
        
        if outputs is not None:
            for detection in outputs:
                category = int(detection.get("category", -1))
                conf = float(detection.get("conf", 0.0))
                
                if conf > 0.30:  
                    coords = detection.get("box", [0, 0, 0, 0])
                    x1, y1 = int(coords[0] * w), int(coords[1] * h)
                    x2, y2 = int(coords[2] * w), int(coords[3] * h)
                    
                    if category == 0:  # Custom Model Phone ID
                        phones.append((x1, y1, x2, y2))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Phone: {conf:.2f}", (x1, max(20, y1 - 5)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    elif category == 1:  # Wrist ID 
                        wx, wy = (x1 + x2) // 2, (y1 + y2) // 2
                        wrists.append((wx, wy))
                        cv2.circle(frame, (wx, wy), 6, (255, 255, 0), -1)

        # Proximity Check
        phone_in_hand = False
        for wx, wy in wrists:
            for x1, y1, x2, y2 in phones:
                if (x1 - 20) <= wx <= (x2 + 20) and (y1 - 20) <= wy <= (y2 + 20):
                    phone_in_hand = True
                    break
            if phone_in_hand:
                break

        if phone_in_hand:
            cv2.rectangle(frame, (10, 10), (280, 60), (0, 0, 255), -1)
            cv2.putText(frame, "PHONE IN HAND", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("IMX500 Hardware Accelerated Tracking", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("\nSafely unlinking camera pipelines...")
    # Explicitly verify the objects exist before allowing garbage collection / unref sequences
    if picam2 is not None:
        try:
            picam2.stop()
            picam2.close()
            print("Picamera2 backend closed safely.")
        except Exception:
            pass
            
    cv2.destroyAllWindows()
    print("System resources freed.")