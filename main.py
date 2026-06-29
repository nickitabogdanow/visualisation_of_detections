from __future__ import annotations

import io
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

import plotly
import plotly.graph_objects as go

from parser import load_csv_files, parse_csv_content
from visualization import FREQ_BIN_MHZ, TIME_BIN_MINUTES, build_heatmap_figure

IS_FROZEN = getattr(sys, "frozen", False)
SOURCE_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", SOURCE_DIR))
APP_DIR = Path(sys.executable).resolve().parent if IS_FROZEN else SOURCE_DIR

DATA_DIR = Path(os.getenv("DATA_DIR", APP_DIR / "data"))
TEMPLATES_DIR = BUNDLE_DIR / "templates"
PLOTLY_JS = BUNDLE_DIR / "plotly.min.js"
if not PLOTLY_JS.exists():
    PLOTLY_JS = Path(plotly.__path__[0]) / "package_data" / "plotly.min.js"

app = FastAPI(title="IDF Visualisation")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

STATIC_DIR = BUNDLE_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/js/plotly.min.js", include_in_schema=False)
async def plotly_js() -> FileResponse:
    return FileResponse(PLOTLY_JS, media_type="application/javascript")


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


def _parse_chart_ranges(
    time_start: str | None = None,
    time_end: str | None = None,
    freq_min: float | None = None,
    freq_max: float | None = None,
) -> dict[str, datetime | float | None]:
    parsed_start = pd.to_datetime(time_start) if time_start else None
    parsed_end = pd.to_datetime(time_end) if time_end else None

    if time_start and pd.isna(parsed_start):
        raise HTTPException(status_code=400, detail="Некорректный формат time_start")
    if time_end and pd.isna(parsed_end):
        raise HTTPException(status_code=400, detail="Некорректный формат time_end")
    if parsed_start is not None and parsed_end is not None and parsed_start > parsed_end:
        raise HTTPException(
            status_code=400,
            detail="Начало временного диапазона должно быть раньше конца",
        )
    if freq_min is not None and freq_max is not None and freq_min > freq_max:
        raise HTTPException(
            status_code=400,
            detail="Минимальная частота должна быть меньше или равна максимальной",
        )

    return {
        "time_start": parsed_start.to_pydatetime() if parsed_start is not None else None,
        "time_end": parsed_end.to_pydatetime() if parsed_end is not None else None,
        "freq_min": freq_min,
        "freq_max": freq_max,
    }


def _build_chart(
    files: list,
    post_number: str,
    time_start: str | None = None,
    time_end: str | None = None,
    freq_min: float | None = None,
    freq_max: float | None = None,
    show_values: bool = True,
    time_bin_minutes: int = TIME_BIN_MINUTES,
    freq_bin_mhz: int = FREQ_BIN_MHZ,
):
    ranges = _parse_chart_ranges(time_start, time_end, freq_min, freq_max)
    try:
        return build_heatmap_figure(
            files,
            post_number,
            **ranges,
            show_values=show_values,
            time_bin_minutes=time_bin_minutes,
            freq_bin_mhz=freq_bin_mhz,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _html_download_response(
    fig,
    post_number: str,
    *,
    show_values: bool = True,
) -> Response:
    trace = fig.data[0]
    z = trace.z
    cell_values = z.tolist() if hasattr(z, "tolist") else z

    if show_values:
        trace.text = cell_values
        trace.texttemplate = "%{text}"
    else:
        trace.text = None
        trace.texttemplate = ""

    buffer = io.StringIO()
    fig.write_html(
        buffer,
        include_plotlyjs=True,
        full_html=True,
        div_id="idf-heatmap",
    )
    html = buffer.getvalue()

    checked = "checked" if show_values else ""
    toolbar = (
        '<div id="idf-toolbar" style="font-family:sans-serif;padding:12px 16px;'
        'background:#f4f6f8;border-bottom:1px solid #d8dee6;">'
        '<label style="cursor:pointer;">'
        f'<input type="checkbox" id="idf-show-values" {checked}> '
        "Показывать значения в ячейках"
        "</label></div>"
    )
    toggle_script = f"""
<script>
(function() {{
  const cellValues = {json.dumps(cell_values)};
  const plotDiv = document.getElementById("idf-heatmap");
  const toggle = document.getElementById("idf-show-values");
  if (!plotDiv || !toggle || typeof Plotly === "undefined") {{
    return;
  }}
  toggle.addEventListener("change", function() {{
    if (toggle.checked) {{
      Plotly.restyle(plotDiv, {{text: [cellValues], texttemplate: ["%{{text}}"]}}, [0]);
    }} else {{
      Plotly.restyle(plotDiv, {{text: [null], texttemplate: [""]}}, [0]);
    }}
  }});
}})();
</script>
"""
    if "<body>" in html:
        html = html.replace("<body>", f"<body>{toolbar}", 1)
    html = html.replace("</body>", toggle_script + "</body>")

    filename = f"post_{post_number}_heatmap.html"
    return Response(
        content=html,
        media_type="text/html; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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
            "local_files_count": len(local_files),
            "posts": posts,
        },
    )


@app.get("/api/files")
async def list_files(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=10, ge=1, le=100),
) -> dict:
    paths = _collect_local_csv_paths()
    total = len(paths)
    start = (page - 1) * per_page
    page_paths = paths[start : start + per_page]
    total_pages = math.ceil(total / per_page) if total else 0
    return {
        "files": [p.name for p in page_paths],
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": total_pages,
    }


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
async def chart_from_local(
    post_number: str,
    time_start: str | None = Query(default=None),
    time_end: str | None = Query(default=None),
    freq_min: float | None = Query(default=None),
    freq_max: float | None = Query(default=None),
    show_values: bool = Query(default=True),
    time_bin_minutes: int = Query(default=TIME_BIN_MINUTES, ge=1),
    freq_bin_mhz: int = Query(default=FREQ_BIN_MHZ, ge=1),
) -> dict:
    paths = _collect_local_csv_paths()
    if not paths:
        raise HTTPException(status_code=404, detail="Нет CSV файлов в папке data/")

    grouped = load_csv_files(paths)
    if post_number not in grouped:
        raise HTTPException(status_code=404, detail=f"Пост {post_number} не найден")

    fig = _build_chart(
        grouped[post_number],
        post_number,
        time_start=time_start,
        time_end=time_end,
        freq_min=freq_min,
        freq_max=freq_max,
        show_values=show_values,
        time_bin_minutes=time_bin_minutes,
        freq_bin_mhz=freq_bin_mhz,
    )
    return json.loads(fig.to_json())


@app.post("/api/export/html")
async def export_chart_html(payload: dict = Body(...)) -> Response:
    post_number = str(payload.get("post_number") or "chart")
    data = payload.get("data")
    layout = payload.get("layout")
    if not data:
        raise HTTPException(status_code=400, detail="Нет данных диаграммы для экспорта")

    fig = go.Figure(data=data, layout=layout or {})
    show_values = bool(payload.get("show_values", True))
    return _html_download_response(fig, post_number, show_values=show_values)


@app.get("/api/chart/{post_number}/html")
async def chart_html_from_local(
    post_number: str,
    time_start: str | None = Query(default=None),
    time_end: str | None = Query(default=None),
    freq_min: float | None = Query(default=None),
    freq_max: float | None = Query(default=None),
    show_values: bool = Query(default=True),
    time_bin_minutes: int = Query(default=TIME_BIN_MINUTES, ge=1),
    freq_bin_mhz: int = Query(default=FREQ_BIN_MHZ, ge=1),
) -> Response:
    paths = _collect_local_csv_paths()
    if not paths:
        raise HTTPException(status_code=404, detail="Нет CSV файлов в папке data/")

    grouped = load_csv_files(paths)
    if post_number not in grouped:
        raise HTTPException(status_code=404, detail=f"Пост {post_number} не найден")

    fig = _build_chart(
        grouped[post_number],
        post_number,
        time_start=time_start,
        time_end=time_end,
        freq_min=freq_min,
        freq_max=freq_max,
        show_values=show_values,
        time_bin_minutes=time_bin_minutes,
        freq_bin_mhz=freq_bin_mhz,
    )
    return _html_download_response(fig, post_number, show_values=show_values)


@app.post("/api/upload/chart")
async def chart_from_upload(
    files: list[UploadFile] = File(...),
    post_number: str | None = Query(default=None),
    time_start: str | None = Query(default=None),
    time_end: str | None = Query(default=None),
    freq_min: float | None = Query(default=None),
    freq_max: float | None = Query(default=None),
    show_values: bool = Query(default=True),
    time_bin_minutes: int = Query(default=TIME_BIN_MINUTES, ge=1),
    freq_bin_mhz: int = Query(default=FREQ_BIN_MHZ, ge=1),
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

    fig = _build_chart(
        grouped[selected],
        selected,
        time_start=time_start,
        time_end=time_end,
        freq_min=freq_min,
        freq_max=freq_max,
        show_values=show_values,
        time_bin_minutes=time_bin_minutes,
        freq_bin_mhz=freq_bin_mhz,
    )
    return {
        "posts": posts,
        "selected_post": selected,
        "chart": json.loads(fig.to_json()),
    }


@app.post("/api/upload/chart/html")
async def chart_html_from_upload(
    files: list[UploadFile] = File(...),
    post_number: str | None = Query(default=None),
    time_start: str | None = Query(default=None),
    time_end: str | None = Query(default=None),
    freq_min: float | None = Query(default=None),
    freq_max: float | None = Query(default=None),
    show_values: bool = Query(default=True),
    time_bin_minutes: int = Query(default=TIME_BIN_MINUTES, ge=1),
    freq_bin_mhz: int = Query(default=FREQ_BIN_MHZ, ge=1),
) -> Response:
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

    fig = _build_chart(
        grouped[selected],
        selected,
        time_start=time_start,
        time_end=time_end,
        freq_min=freq_min,
        freq_max=freq_max,
        show_values=show_values,
        time_bin_minutes=time_bin_minutes,
        freq_bin_mhz=freq_bin_mhz,
    )
    return _html_download_response(fig, selected, show_values=show_values)
