"""Web server: upload a photo, pick a PolyU background and a style, get the souvenir picture.

  uvicorn persona2.server:app --host 0.0.0.0 --port 7860
Set PERSONA_NO_MODEL=1 to start the page without loading the model (layout work, no GPU).
One generation runs at a time; other requests wait for the lock.
"""
from __future__ import annotations

import base64
import io
import os
import threading

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from .analyze import analyze
from .pipeline import ROOT, Persona, load_config

CFG = load_config()
WEB = os.path.join(ROOT, "persona2", "web")
app = FastAPI(title="UltraPersona")
app.mount("/static", StaticFiles(directory=os.path.join(WEB, "static")), name="static")
app.mount("/backgrounds", StaticFiles(directory=os.path.join(ROOT, "backgrounds")), name="backgrounds")

_lock = threading.Lock()
_persona: Persona | None = None


def persona() -> Persona:
    global _persona
    if _persona is None:
        if os.environ.get("PERSONA_NO_MODEL") == "1":
            raise HTTPException(503, "模型没有加载（PERSONA_NO_MODEL=1）。")
        from .qwen_edit import set_vram_cap
        set_vram_cap(CFG["model"]["vram_cap_gb"])
        _persona = Persona(CFG)
    return _persona


@app.on_event("startup")
def _warm():
    if os.environ.get("PERSONA_NO_MODEL") != "1":
        threading.Thread(target=persona, daemon=True).start()


def _read(upload: UploadFile) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(upload.file.read()))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:  # noqa: BLE001
        raise HTTPException(400, "无法读取这张图片，请上传 JPG 或 PNG。")
    if max(img.size) > 2048:
        img.thumbnail((2048, 2048), Image.LANCZOS)
    return img


def _b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"))


@app.get("/api/options")
def options():
    styles_dir = os.path.join(WEB, "static", "styles")
    return {
        "backgrounds": [{"id": k, "name": v.get("name_zh", v["name"]), "max_people": v["max_people"],
                         "url": "/backgrounds/" + os.path.basename(v["file"])}
                        for k, v in CFG["backgrounds"].items()],
        "styles": [{"id": k, "name": v.get("name_zh", v["name"]),
                    "sample": f"/static/styles/{k}.jpg" if os.path.exists(os.path.join(styles_dir, f"{k}.jpg")) else None}
                   for k, v in CFG["styles"].items()],
        "model_ready": _persona is not None,
    }


@app.post("/api/analyze")
def api_analyze(photo: UploadFile = File(...)):
    an = analyze(_read(photo), CFG)
    return {"people": an.people, "status": an.status, "messages": an.messages, "framing": an.framing,
            "allowed_backgrounds": [k for k, v in CFG["backgrounds"].items() if an.people <= v["max_people"]]}


@app.post("/api/generate")
def api_generate(photo: UploadFile = File(...), background: int = Form(...), style: int = Form(...),
                 seed: int = Form(0)):
    if background not in CFG["backgrounds"] or style not in CFG["styles"]:
        raise HTTPException(400, "背景或风格编号不存在。")
    img = _read(photo)
    p = persona()
    with _lock:
        res = p.generate(img, background, style, mode=CFG["styles"][style].get("mode", CFG["model"].get("default_mode", "onestep")), seed=seed)
    if res.image is None:
        return JSONResponse({"ok": False, "messages": res.warnings}, status_code=422)
    return {"ok": True, "image": _b64(res.image), "messages": res.warnings,
            "people": res.analysis["people"],
            "face_similarity": res.faces.get("per_person") if res.faces else None,
            "prompts": res.prompts}
