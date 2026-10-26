"""Automated high-resolution browser screenshot generator for SpamGuard AI."""

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

SCREENSHOTS_DIR = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

def capture_viewport(name: str, width: int, height: int, url: str = "http://localhost:3000"):
    output_path = SCREENSHOTS_DIR / f"{name}.png"
    cmd = [
        CHROME_PATH,
        "--headless",
        "--disable-gpu",
        f"--window-size={width},{height}",
        f"--screenshot={output_path}",
        url,
    ]
    print(f"Capturing {name} ({width}x{height}) -> {output_path.name}...")
    subprocess.run(cmd, check=True)
    if output_path.exists():
        print(f"[OK] Saved {output_path.name} ({output_path.stat().st_size:,} bytes)")

def main():
    print("Capturing multi-breakpoint verified browser screenshots...")
    # 1. Desktop 1920x1080 Hero
    capture_viewport("01_desktop_hero_1920", 1920, 1080, "http://localhost:3000#overview")

    # 2. Laptop 1366x768
    capture_viewport("02_laptop_1366", 1366, 768, "http://localhost:3000#overview")

    # 3. Tablet 768x1024
    capture_viewport("03_tablet_768", 768, 1024, "http://localhost:3000#overview")

    # 4. Mobile 390x844
    capture_viewport("04_mobile_390", 390, 844, "http://localhost:3000#overview")

    # 5. Pipeline section
    capture_viewport("05_pipeline_architecture", 1920, 1080, "http://localhost:3000#pipeline")

    # 6. Single Scanner section
    capture_viewport("06_single_scanner", 1920, 1080, "http://localhost:3000#scanner")

    # 7. Batch CSV section
    capture_viewport("07_batch_classifier", 1920, 1080, "http://localhost:3000#batch")

    # 8. Model Analytics section
    capture_viewport("08_model_analytics", 1920, 1080, "http://localhost:3000#analytics")

    # 9. Methodology & Team section
    capture_viewport("09_methodology_roadmap", 1920, 1080, "http://localhost:3000#methodology")

    print("All multi-device responsive screenshots captured successfully!")

if __name__ == "__main__":
    main()
