# Dental-Restoration-of-Color-Shape-QC
**A lightweight FastAPI web app with a sleek, dark, chat-style UI for dental restoration quality control. Upload a clinical (in-mouth) photo and a lab (bench) photo, draw Tooth and Shade rectangles, and get:

- **ΔE00** color differences (CIELAB, training-free)
- **Shape similarity** (Hu moments, training-free)
- **Sharpness & glare** checks
- **Transparent quality/success score**

**No dataset required.** Built with classical computer vision and color science. As no dataset were found opensource.

## **Demo Video**: https://drive.google.com/file/d/1JLT7zZJuUgj06Da9fba2JCgIc4nbSsJA/view?usp=sharing

**Demo Note**: The demo video shows poor results because the selected photos are not a match. I intended to test with additional case photos, but I did not have enough matching pairs to demonstrate a successful case.

## Features

- **Two-image workflow**: Clinical + Lab photos with ROI drawing (Tooth/Shade)
- **ΔE00 color analysis**: Tooth↔Shade within each photo, Shade↔Shade across photos (lighting check)
- **Shape similarity**: Computed from tooth silhouettes using Hu moments
- **Blur and glare detection**: Laplacian variance for blur, glare masks
- **Clean indigo/violet theme**: Matches chatbot-inspired design
- **Minimal codebase**: Easy to tweak in minutes

## Project Structure

```
qc_app/
├─ app/
│  ├─ main.py          # FastAPI routes + Jinja templates
│  ├─ qc_utils.py      # Image processing (color, shape, penalties)
│  └─ __init__.py
├─ templates/
│  └─ index.html       # Minimal UI
├─ static/
│  ├─ style.css        # Dark theme
│  └─ app.js           # Canvas + ROI drawer (normalized coords)
└─ requirements.txt    # pip-only deps (optional if using conda)
```

## Quick Start

### Option A: Conda (Recommended for Windows)

```bash
# Create and activate environment
conda create -n qcapp python=3.11 -y
conda activate qcapp

# Install compiled packages via conda (no compiler needed)
conda install -c conda-forge numpy=1.26.4 opencv=4.10.0 scikit-image=0.24.0 -y

# Install pure-Python packages via pip
python -m pip install fastapi==0.115.0 uvicorn==0.30.6 jinja2==3.1.4 python-multipart==0.0.9
```

Run the server:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

If activation is unreliable in PowerShell, use the env’s Python directly:

```bash
& "C:\Users\user\miniconda\envs\qcapp\python.exe" -m uvicorn app.main:app --reload
```

#### Optional: `environment.yml`

```yaml
name: qcapp
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy=1.26.4
  - opencv=4.10.0
  - scikit-image=0.24.0
  - pip
  - pip:
      - fastapi==0.115.0
      - uvicorn==0.30.6
      - jinja2==3.1.4
      - python-multipart==0.0.9
```

### Option B: Pure pip venv

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip

# Ensure binary wheels for NumPy on Windows
pip install --only-binary=:all: numpy==1.26.4
pip install opencv-python==4.10.0.84 scikit-image==0.24.0
pip install fastapi==0.115.0 uvicorn==0.30.6 jinja2==3.1.4 python-multipart==0.0.9

uvicorn app.main:app --reload
```

## How to Use

1. Open [http://127.0.0.1:8000](http://127.0.0.1:8000).
2. Upload:
   - **Clinical photo**: In-mouth tooth (or best neighbor) + shade tab in frame.
   - **Lab photo**: Finished restoration + the same shade tab code.
3. Draw two rectangles on each image:
   - **Tooth ROI** (cyan): Flat, mid-facial area; avoid glare.
   - **Shade ROI** (violet): Ceramic face of the same shade tab code.
4. Click **Analyze** to view results below.


## Interpreting Results

### ΔE00 (Lower is Better)
- **≤ 1.0**: Excellent
- **1–2**: Very good
- **2–3.5**: Acceptable
- **3.5–5**: Borderline
- **>5**: Poor

Results include:
- Clinical Tooth ↔ Shade
- Lab Tooth ↔ Shade
- Shade mismatch across photos (large values indicate lighting issues or wrong regions)

### Shape Score
- Calculated from tooth silhouettes using Hu moments (0–1 scale).
- Best consistency with front-ish, mid-facial crops.

### Quality / Success
```python
quality = 0.45*color_score + 0.35*shape_score + 0.10*sharpness_score - 0.05*glare_penalty - 0.05*lighting_mismatch_penalty
success = sigmoid(3*(quality - 0.6))
```
- **High**: ≥0.80
- **Warning**: 0.60–0.79
- **Low**: <0.60

## Under the Hood (No Training Needed)
- **Color**: ROI pixels converted to CIELAB (perceptually uniform), robust means with glare masking, ΔE00 for differences.
- **Shape**: Largest-contour silhouettes + Hu moments distance.
- **Blur**: Laplacian variance.
- **Lighting**: Shade↔Shade ΔE across photos.
- **Scoring**: Transparent formula; tweak weights in `app/qc_utils.py`.

## API & Extensibility
- **GET /**: Serves the UI.
- **POST /analyze**: Accepts `clinical_image`, `lab_image`, `roi_json`.
- Optional health check:
  ```python
  @app.get("/api/health")
  def health(): return {"status": "ok"}
  ```

### Easy Tweaks
- **Theme**: Modify CSS variables in `static/style.css`.
- **Thresholds/Weights**: Edit `compute_metrics()` in `app/qc_utils.py`.
- **Single-Image Mode**: Add a route for one image (Tooth↔Shade ΔE00 only).

## Troubleshooting
- **Analyze button does nothing**: Ensure both Tooth and Shade ROIs are drawn on both images (cyan/purple labels visible).
- **/api/health or favicon.ico 404**: Harmless. Add the health route or `static/favicon.ico` to silence.
- **Windows pip NumPy build errors**: Use Conda or `--only-binary=:all:` for NumPy.
- **Conda run freezes**: Use `--no-capture-output` or call the env’s `python.exe` directly.

## Notes
- Images are processed in-memory and not persisted. Add storage/auth for deployment.
- Ensure your demo video is hosted (e.g., GitHub, YouTube) and update the link above.

## Roadmap
- Single-image quick check
- CSV/PDF export
- OCR shade-code validation
- Auto-suggest ROIs (e.g., SAM) with manual override
- Per-lab calibration (tune weights from outcomes)

## Acknowledgements
- Color science: CIELAB / ΔE00
- Libraries: OpenCV, scikit-image
- Design: Inspired by dark clinical theme**

