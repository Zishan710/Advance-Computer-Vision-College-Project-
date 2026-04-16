"""
Traffic Surveillance System
============================
Detects, tracks, and counts incoming vehicles on the right side of the video.
Categorizes vehicles into: Heavy Vehicle, Passenger Vehicle, 2-Wheeler.
Displays live (current) and total counts on both top corners of the frame.

Author  : Traffic Surveillance System
PEP 8   : Compliant
OOP     : Yes
"""

import warnings
import os

# Suppress numpy/torch version mismatch warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

import cv2
import numpy as np
from collections import defaultdict
from typing import Optional
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_VIDEO = "acv-input_video.mp4"
OUTPUT_VIDEO = "output/output_surveillance.mp4"
MODEL_WEIGHTS = "yolov8n.pt"

# Process every Nth frame to speed up CPU inference.
# Detections are interpolated for skipped frames.
FRAME_SKIP = 2          # process 1 in every 2 frames (~2x speedup)

# Resize frame for inference (smaller = faster). Output is still full-res.
INFER_WIDTH = 640       # yolov8 native input size — no quality loss

# ROI for the RIGHT side of the video (incoming lane).
# Defined as a polygon: (x, y) points.
# Will be scaled at runtime once we know frame dimensions.
# These are fractional values [0.0 – 1.0] relative to frame size.
ROI_POLYGON_FRAC = np.array([
    [0.50, 0.10],   # top-left of right lane
    [1.00, 0.10],   # top-right
    [1.00, 1.00],   # bottom-right
    [0.50, 1.00],   # bottom-left
], dtype=np.float32)

# Counting line: vehicles crossing this horizontal line (y fraction) are counted.
# Placed at ~70 % of frame height so vehicles moving downward cross it.
COUNT_LINE_Y_FRAC = 0.70

# Direction: incoming vehicles on the right side move DOWNWARD (y increases).
# We count a vehicle when its centroid crosses the line moving downward.
DIRECTION = "down"

# YOLO class-id → our category mapping
# COCO classes used: motorcycle(3), car(2), bus(5), truck(7)
HEAVY_IDS = {5, 7}          # bus, truck
PASSENGER_IDS = {2, 3}      # car, auto-rickshaw treated as passenger (class 3 = motorcycle skipped below)
TWO_WHEELER_IDS = {3}       # motorcycle / bike

# Confidence threshold
CONF_THRESHOLD = 0.35

# Colours (BGR)
COLOUR_HEAVY = (0, 0, 255)       # red
COLOUR_PASSENGER = (0, 255, 0)   # green
COLOUR_TWO_WHEELER = (255, 165, 0)  # orange
COLOUR_ROI = (0, 255, 255)       # yellow
COLOUR_LINE = (255, 0, 255)      # magenta

# YOLO class ids we care about
VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def get_category(class_id: int) -> str:
    """Return vehicle category string for a given YOLO class id."""
    if class_id in HEAVY_IDS:
        return "Heavy"
    if class_id in TWO_WHEELER_IDS:
        return "2-Wheeler"
    return "Passenger"


def get_colour(category: str) -> tuple:
    """Return BGR colour for a vehicle category."""
    mapping = {
        "Heavy": COLOUR_HEAVY,
        "Passenger": COLOUR_PASSENGER,
        "2-Wheeler": COLOUR_TWO_WHEELER,
    }
    return mapping.get(category, (255, 255, 255))


def point_in_polygon(point: tuple, polygon: np.ndarray) -> bool:
    """Check whether a point (x, y) lies inside a convex polygon."""
    result = cv2.pointPolygonTest(polygon, point, False)
    return result >= 0


# ---------------------------------------------------------------------------
# Core classes
# ---------------------------------------------------------------------------

class VehicleTracker:
    """
    Maintains per-track state: previous centroid, category, counted flag.
    Uses YOLO's built-in ByteTrack IDs.
    """

    def __init__(self):
        self._tracks: dict[int, dict] = {}

    def update(self, track_id: int, centroid: tuple, category: str) -> None:
        """Update or initialise a track entry."""
        if track_id not in self._tracks:
            self._tracks[track_id] = {
                "prev_y": centroid[1],
                "category": category,
                "counted": False,
            }
        else:
            self._tracks[track_id]["prev_y"] = self._tracks[track_id].get("curr_y", centroid[1])
            self._tracks[track_id]["category"] = category
        self._tracks[track_id]["curr_y"] = centroid[1]

    def should_count(self, track_id: int, line_y: int) -> bool:
        """
        Return True if this track just crossed the counting line moving downward
        and has not been counted yet.
        """
        if track_id not in self._tracks:
            return False
        track = self._tracks[track_id]
        if track["counted"]:
            return False
        prev_y = track.get("prev_y", track["curr_y"])
        curr_y = track["curr_y"]
        # Downward crossing: prev above line, curr at/below line
        if prev_y < line_y <= curr_y:
            track["counted"] = True
            return True
        return False

    def get_category(self, track_id: int) -> str:
        """Return stored category for a track."""
        return self._tracks.get(track_id, {}).get("category", "Passenger")

    def active_ids(self) -> set:
        """Return set of all known track ids."""
        return set(self._tracks.keys())


class CounterBoard:
    """
    Maintains total and current-frame counts per category.
    Renders the HUD overlay on a frame.
    """

    CATEGORIES = ["Heavy", "Passenger", "2-Wheeler"]

    def __init__(self):
        self.total: dict[str, int] = defaultdict(int)
        self._current: dict[str, int] = defaultdict(int)

    def increment_total(self, category: str) -> None:
        """Increment the total count for a category."""
        self.total[category] += 1

    def set_current(self, counts: dict[str, int]) -> None:
        """Set current-frame counts (vehicles visible right now)."""
        self._current = defaultdict(int, counts)

    @property
    def total_all(self) -> int:
        return sum(self.total.values())

    @property
    def current_all(self) -> int:
        return sum(self._current.values())

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Draw count panels on top-left and top-right corners."""
        frame = self._draw_left_panel(frame)
        frame = self._draw_right_panel(frame)
        return frame

    # ------------------------------------------------------------------
    # Private rendering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_bg(frame, x, y, w, h, alpha=0.55):
        """Draw a semi-transparent dark rectangle."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)

    def _draw_left_panel(self, frame: np.ndarray) -> np.ndarray:
        """Top-left: CURRENT vehicle counts."""
        x, y, w, h = 10, 10, 230, 120
        self._draw_bg(frame, x, y, w, h)
        cv2.putText(frame, "CURRENT VEHICLES", (x + 8, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.line(frame, (x + 5, y + 26), (x + w - 5, y + 26), (150, 150, 150), 1)
        row = y + 44
        for cat in self.CATEGORIES:
            colour = get_colour(cat)
            cv2.putText(frame, f"{cat}:", (x + 10, row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, colour, 1)
            cv2.putText(frame, str(self._current[cat]), (x + 165, row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
            row += 22
        cv2.putText(frame, f"Total: {self.current_all}", (x + 10, row + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)
        return frame

    def _draw_right_panel(self, frame: np.ndarray) -> np.ndarray:
        """Top-right: TOTAL vehicle counts."""
        fw = frame.shape[1]
        w, h = 230, 120
        x, y = fw - w - 10, 10
        self._draw_bg(frame, x, y, w, h)
        cv2.putText(frame, "TOTAL VEHICLES", (x + 14, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.line(frame, (x + 5, y + 26), (x + w - 5, y + 26), (150, 150, 150), 1)
        row = y + 44
        for cat in self.CATEGORIES:
            colour = get_colour(cat)
            cv2.putText(frame, f"{cat}:", (x + 10, row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, colour, 1)
            cv2.putText(frame, str(self.total[cat]), (x + 165, row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
            row += 22
        cv2.putText(frame, f"Total: {self.total_all}", (x + 10, row + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)
        return frame


class TrafficSurveillance:
    """
    Main pipeline class.
    Loads video → runs YOLO detection + tracking → applies ROI filter →
    counts incoming (downward-moving) vehicles → writes annotated output.
    """

    def __init__(
        self,
        input_path: str = INPUT_VIDEO,
        output_path: str = OUTPUT_VIDEO,
        model_weights: str = MODEL_WEIGHTS,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.model = YOLO(model_weights)
        self.tracker = VehicleTracker()
        self.board = CounterBoard()

        # Resolved at runtime
        self._roi_polygon: Optional[np.ndarray] = None
        self._count_line_y: int = 0
        self._frame_w: int = 0
        self._frame_h: int = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _init_geometry(self, frame_w: int, frame_h: int) -> None:
        """Compute absolute pixel coordinates for ROI and counting line."""
        self._frame_w = frame_w
        self._frame_h = frame_h
        pts = ROI_POLYGON_FRAC.copy()
        pts[:, 0] *= frame_w
        pts[:, 1] *= frame_h
        self._roi_polygon = pts.astype(np.int32)
        self._count_line_y = int(COUNT_LINE_Y_FRAC * frame_h)

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run detection, tracking, counting and annotation on one frame."""
        results = self.model.track(
            frame,
            persist=True,
            conf=CONF_THRESHOLD,
            classes=list(VEHICLE_CLASS_IDS),
            imgsz=INFER_WIDTH,
            verbose=False,
        )

        current_counts: dict[str, int] = defaultdict(int)
        active_ids_this_frame: set[int] = set()

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                # Skip detections without a track id
                if box.id is None:
                    continue

                track_id = int(box.id.item())
                class_id = int(box.cls.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                category = get_category(class_id)

                # --- ROI filter: only process vehicles inside the right-lane ROI ---
                if not point_in_polygon((cx, cy), self._roi_polygon):
                    continue

                self.tracker.update(track_id, (cx, cy), category)
                active_ids_this_frame.add(track_id)
                current_counts[category] += 1

                # --- Counting line crossing ---
                if self.tracker.should_count(track_id, self._count_line_y):
                    self.board.increment_total(category)

                # --- Draw bounding box ---
                colour = get_colour(category)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                label = f"#{track_id} {category}"
                label_y = y1 - 6 if y1 > 20 else y1 + 16
                cv2.putText(frame, label, (x1, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1)

        # Update current counts
        self.board.set_current(current_counts)

        # --- Draw ROI polygon ---
        cv2.polylines(frame, [self._roi_polygon], isClosed=True,
                      color=COLOUR_ROI, thickness=2)

        # --- Draw counting line ---
        cv2.line(frame,
                 (self._roi_polygon[:, 0].min(), self._count_line_y),
                 (self._frame_w, self._count_line_y),
                 COLOUR_LINE, 2)
        cv2.putText(frame, "COUNT LINE", (self._frame_w // 2 + 10, self._count_line_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOUR_LINE, 1)

        # --- HUD overlay ---
        frame = self.board.render(frame)
        return frame

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full surveillance pipeline."""
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.input_path}")

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._init_geometry(frame_w, frame_h)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.output_path, fourcc, fps, (frame_w, frame_h))

        print(f"[INFO] Processing {total_frames} frames @ {fps:.1f} fps ...")
        print(f"[INFO] ROI polygon (px): {self._roi_polygon.tolist()}")
        print(f"[INFO] Count line Y: {self._count_line_y}px")
        print(f"[INFO] Frame skip: {FRAME_SKIP} | Infer width: {INFER_WIDTH}px")

        frame_idx = 0
        last_annotated = None  # reuse last annotation for skipped frames

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_SKIP == 0:
                annotated = self._process_frame(frame)
                last_annotated = annotated
            else:
                # For skipped frames: draw HUD on raw frame without re-running YOLO
                annotated = self.board.render(frame.copy())
                # Re-draw ROI and count line
                cv2.polylines(annotated, [self._roi_polygon], isClosed=True,
                              color=COLOUR_ROI, thickness=2)
                cv2.line(annotated,
                         (self._roi_polygon[:, 0].min(), self._count_line_y),
                         (self._frame_w, self._count_line_y),
                         COLOUR_LINE, 2)

            writer.write(annotated)

            frame_idx += 1
            if frame_idx % 100 == 0:
                pct = frame_idx / total_frames * 100 if total_frames else 0
                print(f"  Frame {frame_idx}/{total_frames} ({pct:.1f}%) | "
                      f"Total: {self.board.total_all} | "
                      f"Heavy={self.board.total['Heavy']} "
                      f"Passenger={self.board.total['Passenger']} "
                      f"2-Wheeler={self.board.total['2-Wheeler']}")

        cap.release()
        writer.release()

        print("\n[DONE] Final counts:")
        print(f"  Heavy Vehicle  : {self.board.total['Heavy']}")
        print(f"  Passenger      : {self.board.total['Passenger']}")
        print(f"  2-Wheeler      : {self.board.total['2-Wheeler']}")
        print(f"  TOTAL          : {self.board.total_all}")
        print(f"[INFO] Output saved → {self.output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pipeline = TrafficSurveillance(
        input_path=INPUT_VIDEO,
        output_path=OUTPUT_VIDEO,
        model_weights=MODEL_WEIGHTS,
    )
    pipeline.run()
