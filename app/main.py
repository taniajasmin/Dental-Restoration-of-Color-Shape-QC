import json
from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .qc_utils import read_image_from_bytes, compute_metrics

app = FastAPI(title="Dental QC (No-Training)")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": None})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    clinical_image: UploadFile,
    lab_image: UploadFile,
    roi_json: str = Form(...)
):
    try:
        roi = json.loads(roi_json)
    except Exception:
        return templates.TemplateResponse("index.html",
            {"request": request, "error": "Invalid ROI payload", "results": None}
        )

    img_c = read_image_from_bytes(await clinical_image.read())
    img_l = read_image_from_bytes(await lab_image.read())

    results = compute_metrics(img_c, img_l, roi)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": results,
        "roi": json.dumps(roi),
        "clinical_name": clinical_image.filename,
        "lab_name": lab_image.filename
    })
