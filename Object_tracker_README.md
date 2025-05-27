# Video Object Tracker with Scene Change Detection

This Python script lets you track objects in video footage using OpenCV’s tracking algorithms. 
It also detects scene changes using SSIM (Structural Similarity Index) to automatically reset tracking when the scene cuts.

## 🚀 Features

- Multiple tracking algorithms to choose from, each suited to different scenarios:
  - **BOOSTING:** Basic AdaBoost-based tracker; slower and less accurate, but good for learning.
  - **MIL:** Improved robustness over BOOSTING; better with appearance changes.
  - **KCF:** Fast and accurate, suitable for moderate object motion.
  - **TLD:** Combines tracking and detection; recovers from failures but may give false positives.
  - **MEDIANFLOW:** Precise with slow, predictable motion; struggles with fast moves.
  - **MOSSE:** Extremely fast and lightweight; great for real-time use on low-end hardware.
  - **CSRT:** Most accurate and robust; handles scale and rotation changes well but slower.

- Detects scene cuts by comparing consecutive frames, so the tracker can reset when the video content changes abruptly.
- Real-time FPS display for performance monitoring.
- Works with both video files and live webcam input.

## 📦 Requirements

- Python 3.x
- OpenCV (`opencv-contrib-python`)
- scikit-image
- NumPy

Install dependencies via:

```bash
pip install opencv-contrib-python scikit-image numpy
