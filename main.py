from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from parser import load_csv_files, parse_csv_content
from visualization import build_heatmap_figure

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

app = FastAPI(title="IDF Visualisation")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

if (BASE_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def _collect_local_csv_paths() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("*.csv"))


def _group_uploads(files: list[UploadFile]) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for upload in files:
        content = upload.file.read()
        try:
            parsed = parse_csv_content(content, source=upload.filename or "upload.csv")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        post = parsed.metadata.post_number or "unknown"
        grouped.setdefault(post, []).append(parsed)
    return grouped


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    local_files = _collect_local_csv_paths()
    posts: list[str] = []
    if local_files:
        posts = sorted(load_csv_files(local_files).keys())
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "local_files": [p.name for p in local_files],
            "posts": posts,
        },
    )


@app.get("/api/posts")
async def list_posts() -> dict:
    paths = _collect_local_csv_paths()
    if not paths:
        return {"posts": [], "files": []}
    grouped = load_csv_files(paths)
    return {
        "posts": sorted(grouped.keys()),
        "files": [p.name for p in paths],
    }


@app.get("/api/chart/{post_number}")
async def chart_from_local(post_number: str) -> dict:
    paths = _collect_local_csv_paths()
    if not paths:
        raise HTTPException(status_code=404, detail="Нет CSV файлов в папке data/")

    grouped = load_csv_files(paths)
    if post_number not in grouped:
        raise HTTPException(status_code=404, detail=f"Пост {post_number} не найден")

    fig = build_heatmap_figure(grouped[post_number], post_number)
    return json.loads(fig.to_json())


@app.post("/api/upload/chart")
async def chart_from_upload(
    files: list[UploadFile] = File(...),
    post_number: str | None = Query(default=None),
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="Загрузите хотя бы один CSV файл")

    grouped = _group_uploads(files)
    posts = sorted(grouped.keys())

    selected = post_number or posts[0]
    if selected not in grouped:
        raise HTTPException(
            status_code=404,
            detail=f"Пост {selected} не найден. Доступные: {', '.join(posts)}",
        )

    fig = build_heatmap_figure(grouped[selected], selected)
    return {
        "posts": posts,
        "selected_post": selected,
        "chart": json.loads(fig.to_json()),
    }
