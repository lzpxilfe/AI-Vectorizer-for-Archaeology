# AI Vectorizer for Archaeology

QGIS Plugin for AI-assisted contour digitizing from historical maps.

## Features
- **Smart Trace Tool**: Click two points to automatically trace contour lines between them using A* pathfinding.
- **AI Modes**:
  - **Lite Mode**: Uses OpenCV Canny Edge Detection (CPU, No GPU required).
  - **Standard/Pro Mode**: (Coming in Phase 2) SAM-based segmentation.

## Installation

### 1. Developer Install
Because this is a development version, you need to link the plugin to your QGIS profile.

**Windows (PowerShell):**
```powershell
$src = "C:\Users\nuri9\AI-Vectorizer-for-Archaeology\ai_vectorizer"
$dest = "$env:APPDATA\QGIS\QGIS3\profiles\default\python\plugins\ai_vectorizer"
New-Item -ItemType Junction -Path $dest -Target $src
```

### 2. Dependencies
Open QGIS Python Console and install requirements:
```python
import pip
pip.main(['install', 'opencv-python-headless', 'scikit-image'])
```
Or install via OS terminal.

## Usage
1. Open QGIS and load your historical map raster (e.g., `174 청양.jpg`).
2. Create a new Vector Layer (LineString) for the output.
3. Open **AI Vectorizer** from the Toolbar or Vector Menu.
4. Select your Raster and Vector layers.
5. Choose **Lite (OpenCV)** model.
6. Click **Activate Smart Trace Tool**.
7. Click a start point on a contour line, then a second point along the line.
8. The tool will calculate the path and add it to the vector layer.
