"""
utils/logger.py — Violation event logging
"""

import csv
import os
import time
from datetime import datetime


class ViolationLogger:
    def __init__(self, log_dir="results"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.events = []
        self.session_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log(self, violation_type, confidence, frame_num):
        self.events.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "frame": frame_num,
            "violation": violation_type,
            "confidence": round(confidence, 3),
        })

    def save_report(self):
        if not self.events:
            print("[LOG] No violations detected. No report saved.")
            return

        report_path = os.path.join(self.log_dir, f"report_{int(time.time())}.csv")
        with open(report_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "frame", "violation", "confidence"])
            writer.writeheader()
            writer.writerows(self.events)

        print(f"[LOG] Report saved to {report_path}  ({len(self.events)} violations)")
        return report_path
