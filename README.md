# Driver Attentiveness Monitor

Production-style Raspberry Pi 5 driver attentiveness monitor using Picamera2, Raspberry Pi AI Camera, YOLOv8 ONNX inference, MediaPipe Pose, and MediaPipe Face Mesh.

## Hardware Target

- Raspberry Pi 5
- Raspberry Pi AI Camera, Sony IMX500
- Raspberry Pi OS Bookworm 64-bit
- Python 3.11+

## Folder Layout

```text
project/
├── main.py
├── config.py
├── detector.py
├── pose.py
├── face.py
├── scoring.py
├── drawing.py
├── utils.py
├── requirements.txt
├── README.md
└── model/
    └── phone_detector.onnx
```

## Setup

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-venv libcap-dev
cd project
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For best Pi stability, prefer the distro `python3-picamera2` package and create the venv with `--system-site-packages`, as shown above.

## Model

Place your YOLOv8n ONNX model at:

```text
model/phone_detector.onnx
```

The detector expects a YOLOv8-style output tensor in either `[1, 84, N]`, `[1, N, 84]`, or `[1, N, 6]` format. COCO class `0` is treated as `person`; COCO class `67` is treated as `phone`.

Example export:

```bash
yolo export model=yolov8n.pt format=onnx imgsz=640 opset=12 simplify=True
cp yolov8n.onnx model/phone_detector.onnx
```

If you train a custom two-class model, map class `0` to `person` and class `1` to `phone`, or adjust `COCO_NAMES` in `detector.py`.

## Run

Picamera2:

```bash
python main.py
```

USB camera fallback:

```bash
python main.py --usb --usb-index 0
```

Headless smoke test:

```bash
python main.py --no-window
```

Press `q` or `Esc` to exit.

## Scoring Rules

The score starts at 100.

- Phone held:
  - `<1s`: `0`
  - `1-3s`: `-10`
  - `3-5s`: `-25`
  - `>5s`: `-40`
- Eyes away:
  - `<2s`: `0`
  - `2-4s`: `-15`
  - `>4s`: `-30`
- Head away:
  - `>2s`: `-20`

Phone-in-hand requires a phone bounding-box center to remain near either wrist for at least `0.5s`.

## ONNX Inference Notes

The ONNX path uses `onnxruntime` CPU execution. On Raspberry Pi 5, keep the input at `640x640` or lower, use YOLOv8n, and avoid running extra high-resolution post-processing. The application captures at `640x480`, letterboxes to the model input, decodes boxes, filters `person` and `phone`, then applies NMS.

If CPU FPS is too low:

- Export at `imgsz=416` or `imgsz=320` and update `YOLO_INPUT_SIZE`.
- Use a custom model trained only for `person` and `phone`.
- Process detection every second frame while keeping MediaPipe every frame.
- Disable Face Mesh refinement only if iris-based gaze is not required.

## IMX500 ONNX to RPK Overview

For Raspberry Pi AI Camera deployment, the practical path is:

1. Train or choose a compact YOLOv8n phone/person model.
2. Export to ONNX with static input shape and supported operators.
3. Quantize and compile with Sony/Raspberry Pi IMX500 tooling to produce an `.rpk`.
4. Load the `.rpk` through the Picamera2 IMX500 examples/pipeline.
5. Replace `YoloV8OnnxDetector.detect()` with parsed IMX500 metadata detections, keeping the rest of the scoring, pose, face, and drawing modules unchanged.

Compatibility notes:

- Use static input resolution.
- Prefer standard convolution, activation, resize, and detection-head patterns known to be accepted by the IMX500 converter.
- Dynamic shapes, unsupported post-processing ops, and custom layers usually need to be removed from the graph.
- Keep NMS/post-processing on the CPU unless your compiled model/package explicitly supports it.
- Validate class ordering after conversion; the scoring logic assumes labels normalize to `person` and `phone`.

## Clean Shutdown

`main.py` closes MediaPipe objects, stops Picamera2, releases OpenCV capture objects, closes the optional buzzer, and destroys OpenCV windows in a `finally` block to reduce shutdown warnings and GLib-related camera cleanup issues.
