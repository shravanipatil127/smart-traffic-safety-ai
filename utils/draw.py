"""
utils/draw.py — Annotation and overlay helpers
"""

import cv2
import numpy as np


def draw_detections(frame, detections):
    """Draw bounding boxes and labels for all detections."""
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        conf = det["conf"]
        color = det["color"]
        is_violation = det["violation"]

        # Draw bounding box
        thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Violation glow effect (extra border)
        if is_violation:
            cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), color, 1)

        # Label background
        display_text = f"{label.replace('_', ' ').upper()}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y = max(y1 - 4, th + 6)
        cv2.rectangle(frame, (x1, label_y - th - 6), (x1 + tw + 8, label_y + 2), color, -1)

        # Label text
        cv2.putText(frame, display_text, (x1 + 4, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Warning icon for violations
        if is_violation:
            icon_x, icon_y = x2 - 28, y1 + 4
            pts = np.array([[icon_x + 10, icon_y],
                            [icon_x, icon_y + 18],
                            [icon_x + 20, icon_y + 18]], np.int32)
            cv2.fillPoly(frame, [pts], (0, 50, 220))
            cv2.putText(frame, "!", (icon_x + 8, icon_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def draw_stats(frame, frame_num, violations, fps):
    """Draw stats overlay panel in top-left corner."""
    h, w = frame.shape[:2]
    panel_w, panel_h = 260, 90
    overlay = frame.copy()

    # Semi-transparent panel background
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Title
    cv2.putText(frame, "ROAD SAFETY DETECTOR", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Stats
    cv2.putText(frame, f"Frame  : {frame_num}", (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

    viol_color = (0, 60, 230) if violations > 0 else (0, 200, 80)
    cv2.putText(frame, f"Violations: {violations}", (10, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, viol_color, 1, cv2.LINE_AA)

    cv2.putText(frame, f"FPS    : {fps:.1f}", (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

    # Live indicator dot
    cv2.circle(frame, (w - 20, 20), 7, (0, 200, 80), -1)
    cv2.putText(frame, "LIVE", (w - 50, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 80), 1, cv2.LINE_AA)

    return frame
