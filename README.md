# Hough Transform Detection Toolkit

This repository contains six Python scripts demonstrating edge-based feature detection using Hough methods and interactive parameter tuning in OpenCV.

```
├── detect_lines_hough.py        # 1. Standard HoughLines() for infinite line detection
├── detect_lines_houghp.py       # 2. Probabilistic HoughLinesP() for line segments
├── detect_circles_hough.py      # 3. HoughCircles() for circle detection
├── hough_trackbar.py            # 4. Interactive trackbar GUI for Hough parameter tuning
├── lane_detection_hough.py      # 5. Road lane detection in video using HoughLinesP
└── canny_hough_demo.py          # 6. Canny + Hough demo (standard & probabilistic)
```

## Requirements

- Python 3.6 or higher
- OpenCV (`opencv-python`)
- NumPy

Install via pip:

```bash
pip install opencv-python numpy
```

---

## Usage

### 1. Standard Hough Lines
**`detect_lines_hough.py`**
Detects infinite lines via the standard Hough Transform.
```bash
python detect_lines_hough.py image.jpg \
  [--canny1 50] [--canny2 150] \
  [--rho 1.0] [--theta 0.01745] [--threshold 200] \
  [-o output.jpg]
```

### 2. Probabilistic Hough Lines
**`detect_lines_houghp.py`**
Detects line segments via the probabilistic Hough Transform.
```bash
python detect_lines_houghp.py image.jpg \
  [--canny1 50] [--canny2 150] \
  [--rho 1.0] [--theta 0.01745] [--threshold 50] \
  [--minlen 50] [--maxgap 10] \
  [-o output.jpg]
```

### 3. Circle Detection
**`detect_circles_hough.py`**
Finds circles with the Hough Circle Transform.
```bash
python detect_circles_hough.py image.jpg \
  [--dp 1.2] [--min_dist 20] [--param1 50] [--param2 30] \
  [--min_radius 0] [--max_radius 0] \
  [-o output.jpg]
```

### 4. Interactive Hough Tuning
**`hough_trackbar.py`**
Adjust Canny and HoughLinesP parameters in real time:
```bash
python hough_trackbar.py image.jpg
```

### 5. Lane Detection in Video
**`lane_detection_hough.py`**
Detects road lanes in a video using Canny + ROI + HoughLinesP:
```bash
python lane_detection_hough.py video.mp4 \
  [--canny1 50] [--canny2 150] \
  [--rho 2] [--theta 0.01745] [--threshold 100] \
  [--minlen 40] [--maxgap 5] \
  [-o lanes.mp4]
```

### 6. Canny + Hough Demo
**`canny_hough_demo.py`**
Shows standard and probabilistic Hough after Canny on a single image:
```bash
python canny_hough_demo.py image.jpg \
  [--canny1 50] [--canny2 150] \
  [--rho 1.0] [--theta 0.01745] [--threshold 100] \
  [--minlen 50] [--maxgap 10]
```

---

## Notes
- Press **ESC** or **q** to exit any display window.
- Tweak the parameters to suit your specific images or videos.

## License

MIT License
