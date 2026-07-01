"""Driver attentiveness monitor for Raspberry Pi 5 + AI Camera."""

from __future__ import annotations

import argparse
import time
from typing import Optional

import cv2

import config
from detector import YoloV8OnnxDetector
from drawing import draw_detections, draw_face, draw_hud, draw_pose
from face import FaceAnalyzer
from pose import PoseTracker
from scoring import AttentivenessScorer
from utils import FPSCounter, bbox_center, nearest_distance

try:
    from picamera2 import Picamera2
    PICAMERA2_IMPORT_ERROR = None
except Exception as exc:
    Picamera2 = None
    PICAMERA2_IMPORT_ERROR = exc


class CameraSource:
    """Raspberry Pi AI Camera source through Picamera2."""

    def __init__(self) -> None:
        self.picam = None
        self.source_name = "Pi AI Camera / Picamera2"
        if Picamera2 is None:
            raise RuntimeError(
                "Picamera2 could not be imported by this Python interpreter. "
                f"Import error: {PICAMERA2_IMPORT_ERROR!r}. "
                "If Picamera2 is installed with apt, run inside a venv created with "
                "`python3 -m venv .venv --system-site-packages`, or run with system Python."
            )

        self.picam = Picamera2()
        self._configure_picamera2()
        self.picam.start()
        time.sleep(0.8)

        self._validate_stream()

    def _configure_picamera2(self) -> None:
        # The preview configuration path matches Raspberry Pi's common examples
        # and reliably returns RGB888 arrays from capture_array().
        try:
            self.picam.preview_configuration.main.size = (config.FRAME_WIDTH, config.FRAME_HEIGHT)
            self.picam.preview_configuration.main.format = "RGB888"
            self.picam.preview_configuration.align()
            self.picam.configure("preview")
            return
        except Exception as exc:
            print(f"Picamera2 preview configuration failed, trying video configuration: {exc}")

        cfg = self.picam.create_video_configuration(
            main={"size": (config.FRAME_WIDTH, config.FRAME_HEIGHT), "format": "RGB888"},
            controls={"FrameRate": config.CAMERA_FPS},
        )
        self.picam.configure(cfg)

    def read(self):
        try:
            rgb = self.picam.capture_array()
        except TypeError:
            rgb = self.picam.capture_array("main")
        if rgb is None or rgb.size == 0:
            raise RuntimeError("Picamera2 returned an empty frame")
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _validate_stream(self) -> None:
        for _ in range(10):
            try:
                frame = self.read()
            except Exception:
                time.sleep(0.05)
                continue
            if frame is not None and frame.size > 0:
                print(f"Camera ready: {self.source_name} ({frame.shape[1]}x{frame.shape[0]})")
                return
        raise RuntimeError(f"{self.source_name} did not produce a valid startup frame")

    def close(self) -> None:
        if self.picam is not None:
            self.picam.stop()
            self.picam.close()


class OptionalBuzzer:
    def __init__(self) -> None:
        self.buzzer = None
        if not config.ENABLE_BUZZER:
            return
        try:
            from gpiozero import Buzzer

            self.buzzer = Buzzer(config.BUZZER_GPIO_PIN)
        except Exception as exc:
            print(f"Buzzer disabled: {exc}")

    def update(self, score: int) -> None:
        if self.buzzer is None:
            return
        if score < config.BUZZER_SCORE_THRESHOLD:
            self.buzzer.on()
        else:
            self.buzzer.off()

    def close(self) -> None:
        if self.buzzer is not None:
            self.buzzer.off()
            self.buzzer.close()


def phone_in_hand(detections, pose) -> bool:
    phones = [d for d in detections if d.label == "phone"]
    wrists = pose.wrists
    if not phones or not wrists:
        return False
    for phone in phones:
        distance = nearest_distance(bbox_center(phone.box), wrists)
        if distance is not None and distance <= config.PHONE_WRIST_DISTANCE_PX:
            return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(description="Driver attentiveness monitor")
    parser.add_argument("--no-window", action="store_true", help="Run headless without cv2.imshow")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    camera: Optional[CameraSource] = None
    pose_tracker: Optional[PoseTracker] = None
    face_analyzer: Optional[FaceAnalyzer] = None
    buzzer = OptionalBuzzer()

    try:
        camera = CameraSource()
        detector = YoloV8OnnxDetector()
        pose_tracker = PoseTracker()
        face_analyzer = FaceAnalyzer()
        scorer = AttentivenessScorer()
        fps_counter = FPSCounter()

        while True:
            frame = camera.read()
            detections = detector.detect(frame)
            pose = pose_tracker.process(frame)
            face = face_analyzer.process(frame)

            phone_state = phone_in_hand(detections, pose)
            score = scorer.update(phone_state, face.eyes_away, face.head_away)
            buzzer.update(score.score)
            fps = fps_counter.update()

            draw_detections(frame, detections)
            draw_pose(frame, pose)
            draw_face(frame, face)
            draw_hud(frame, score, fps, detector.enabled)

            if not args.no_window:
                cv2.imshow(config.DISPLAY_WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        buzzer.close()
        if face_analyzer is not None:
            face_analyzer.close()
        if pose_tracker is not None:
            pose_tracker.close()
        if camera is not None:
            camera.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
