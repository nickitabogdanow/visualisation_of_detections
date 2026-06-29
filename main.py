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
from fastapi.routing import APIRoute
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from starlette.responses import Response as StarletteResponse

import plotly
import plotly.graph_objects as go

from parser import ParsedCsv, load_csv_files, parse_csv_content
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

MAX_UPLOAD_FILES = 5_000


class LargeUploadRoute(APIRoute):
    def get_route_handler(self):
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> StarletteResponse:
            original_get_form = request._get_form

            async def form_with_file_limit(
                *,
                max_files: int | float = MAX_UPLOAD_FILES,
                max_fields: int | float = MAX_UPLOAD_FILES,
                max_part_size: int = 1024 * 1024,
            ):
                return await original_get_form(
                    max_files=max_files,
                    max_fields=max_fields,
                    max_part_size=max_part_size,
                )

            request.form = form_with_file_limit
            return await original_route_handler(request)

        return custom_route_handler


app = FastAPI(title="IDF Visualisation")
app.router.route_class = LargeUploadRoute
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

STATIC_DIR = BUNDLE_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

CsvSignature = tuple[tuple[str, int, int], ...]

_local_csv_cache_signature: CsvSignature | None = None
_local_csv_cache: dict[str, list[ParsedCsv]] | None = None


@app.get("/js/plotly.min.js", include_in_schema=False)
async def plotly_js() -> FileResponse:
    return FileResponse(PLOTLY_JS, media_type="application/javascript")


def _collect_local_csv_paths() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("*.csv"))


def _local_csv_signature(paths: list[Path]) -> CsvSignature:
    return tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in paths
    )


def _load_local_csv_files(paths: list[Path]) -> dict[str, list[ParsedCsv]]:
    global _local_csv_cache, _local_csv_cache_signature

    signature = _local_csv_signature(paths)
    if _local_csv_cache is not None and signature == _local_csv_cache_signature:
        return _local_csv_cache

    _local_csv_cache = load_csv_files(paths)
    _local_csv_cache_signature = signature
    return _local_csv_cache


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
    hide_zero_rows: bool = False,
    hide_zero_columns: bool = False,
) -> Response:
    trace = fig.data[0]
    z = trace.z
    cell_values = z.tolist() if hasattr(z, "tolist") else z

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

    values_checked = "checked" if show_values else ""
    rows_checked = "checked" if hide_zero_rows else ""
    columns_checked = "checked" if hide_zero_columns else ""
    toolbar = (
        '<div id="idf-toolbar" style="font-family:sans-serif;padding:12px 16px;'
        'background:#f4f6f8;border-bottom:1px solid #d8dee6;">'
        '<label style="cursor:pointer;margin-right:16px;">'
        f'<input type="checkbox" id="idf-show-values" {values_checked}> '
        "Показывать значения в ячейках"
        "</label>"
        '<label style="cursor:pointer;margin-right:16px;">'
        f'<input type="checkbox" id="idf-hide-zero-rows" {rows_checked}> '
        "Скрывать нулевые частоты"
        "</label>"
        '<label style="cursor:pointer;">'
        f'<input type="checkbox" id="idf-hide-zero-columns" {columns_checked}> '
        "Скрывать нулевое время"
        "</label></div>"
    )
    toggle_script = f"""
<script>
(function() {{
  const cellValues = {json.dumps(cell_values)};
  const plotDiv = document.getElementById("idf-heatmap");
  const valuesToggle = document.getElementById("idf-show-values");
  const rowsToggle = document.getElementById("idf-hide-zero-rows");
  const columnsToggle = document.getElementById("idf-hide-zero-columns");
  if (!plotDiv || !valuesToggle || !rowsToggle || !columnsToggle || typeof Plotly === "undefined") {{
    return;
  }}

  const CELL_WIDTH_PX = 56;
  const CELL_HEIGHT_PX = 28;
  const CHART_EXTRA_WIDTH_PX = 160;
  const CHART_EXTRA_HEIGHT_PX = 200;

  const baseTrace = {{
    ...plotDiv.data[0],
    x: [...(plotDiv.data[0].x || [])],
    y: [...(plotDiv.data[0].y || [])],
    z: cellValues,
    text: null,
    texttemplate: "",
  }};
  const baseLayout = {{ ...plotDiv.layout }};

  function isNonZero(value) {{
    return Number(value) !== 0;
  }}

  function getVisibleCategoryCount(axisRange, totalCount) {{
    if (!Array.isArray(axisRange) || axisRange.length < 2) {{
      return totalCount;
    }}

    const start = Math.max(0, Math.floor(Math.min(axisRange[0], axisRange[1]) + 0.5));
    const end = Math.min(totalCount - 1, Math.ceil(Math.max(axisRange[0], axisRange[1]) - 0.5));
    return Math.max(1, end - start + 1);
  }}

  function buildSizedLayout(trace, sourceLayout, axisRanges = {{}}) {{
    const hasXAxisRange = Object.prototype.hasOwnProperty.call(axisRanges, "xaxis");
    const hasYAxisRange = Object.prototype.hasOwnProperty.call(axisRanges, "yaxis");
    const xRange = hasXAxisRange ? axisRanges.xaxis : (sourceLayout.xaxis ? sourceLayout.xaxis.range : null);
    const yRange = hasYAxisRange ? axisRanges.yaxis : (sourceLayout.yaxis ? sourceLayout.yaxis.range : null);
    const xCount = getVisibleCategoryCount(xRange, trace.x ? trace.x.length : 0);
    const yCount = getVisibleCategoryCount(yRange, trace.y ? trace.y.length : 0);

    return {{
      ...sourceLayout,
      width: Math.max(1, xCount) * CELL_WIDTH_PX + CHART_EXTRA_WIDTH_PX,
      height: Math.max(1, yCount) * CELL_HEIGHT_PX + CHART_EXTRA_HEIGHT_PX,
    }};
  }}

  function buildFilteredTrace() {{
    const baseX = baseTrace.x || [];
    const baseY = baseTrace.y || [];
    const baseZ = baseTrace.z || [];

    const rowIndices = baseY
      .map((_, index) => index)
      .filter((index) => !rowsToggle.checked || (baseZ[index] || []).some(isNonZero));
    const colIndices = baseX
      .map((_, index) => index)
      .filter((index) => (
        !columnsToggle.checked || baseZ.some((row) => isNonZero((row || [])[index]))
      ));

    const z = rowIndices.map((rowIndex) => (
      colIndices.map((colIndex) => (baseZ[rowIndex] || [])[colIndex] || 0)
    ));

    return {{
      ...baseTrace,
      x: colIndices.map((index) => baseX[index]),
      y: rowIndices.map((index) => baseY[index]),
      z,
      text: valuesToggle.checked ? z : null,
      texttemplate: valuesToggle.checked ? "%{{text}}" : "",
      textfont: valuesToggle.checked ? {{ size: 10 }} : undefined,
    }};
  }}

  function applyOptions() {{
    const trace = buildFilteredTrace();
    Plotly.react(plotDiv, [trace], buildSizedLayout(trace, baseLayout));
  }}

  let isResizing = false;
  let resizeTimer = null;
  if (typeof plotDiv.on === "function") {{
    plotDiv.on("plotly_relayout", function(eventData) {{
      if (isResizing) {{
        return;
      }}

      const xRange = eventData["xaxis.range"] || (
        eventData["xaxis.range[0]"] != null && eventData["xaxis.range[1]"] != null
          ? [eventData["xaxis.range[0]"], eventData["xaxis.range[1]"]]
          : null
      );
      const yRange = eventData["yaxis.range"] || (
        eventData["yaxis.range[0]"] != null && eventData["yaxis.range[1]"] != null
          ? [eventData["yaxis.range[0]"], eventData["yaxis.range[1]"]]
          : null
      );

      if (!xRange && !yRange && !eventData["xaxis.autorange"] && !eventData["yaxis.autorange"]) {{
        return;
      }}

      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function() {{
        const trace = plotDiv.data[0];
        const sizedLayout = buildSizedLayout(trace, plotDiv.layout, {{
          xaxis: eventData["xaxis.autorange"] ? null : xRange,
          yaxis: eventData["yaxis.autorange"] ? null : yRange,
        }});
        isResizing = true;
        Plotly.relayout(plotDiv, {{
          width: sizedLayout.width,
          height: sizedLayout.height,
        }}).finally(function() {{
          isResizing = false;
        }});
      }}, 120);
    }});
  }}

  valuesToggle.addEventListener("change", applyOptions);
  rowsToggle.addEventListener("change", applyOptions);
  columnsToggle.addEventListener("change", applyOptions);
  applyOptions();
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
        posts = sorted(_load_local_csv_files(local_files).keys())
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
    grouped = _load_local_csv_files(paths)
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
) -> Response:
    paths = _collect_local_csv_paths()
    if not paths:
        raise HTTPException(status_code=404, detail="Нет CSV файлов в папке data/")

    grouped = _load_local_csv_files(paths)
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
    return Response(content=fig.to_json(), media_type="application/json")


@app.post("/api/export/html")
async def export_chart_html(payload: dict = Body(...)) -> Response:
    post_number = str(payload.get("post_number") or "chart")
    data = payload.get("data")
    layout = payload.get("layout")
    if not data:
        raise HTTPException(status_code=400, detail="Нет данных диаграммы для экспорта")

    fig = go.Figure(data=data, layout=layout or {})
    show_values = bool(payload.get("show_values", True))
    hide_zero_rows = bool(payload.get("hide_zero_rows", False))
    hide_zero_columns = bool(payload.get("hide_zero_columns", False))
    return _html_download_response(
        fig,
        post_number,
        show_values=show_values,
        hide_zero_rows=hide_zero_rows,
        hide_zero_columns=hide_zero_columns,
    )


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

    grouped = _load_local_csv_files(paths)
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
    return _html_download_response(
        fig,
        post_number,
        show_values=show_values,
        hide_zero_rows=hide_zero_rows,
        hide_zero_columns=hide_zero_columns,
    )


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
) -> Response:
    if not files:
        raise HTTPException(status_code=400, detail="Загрузите хотя бы один CSV файл")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Можно загрузить не больше {MAX_UPLOAD_FILES} файлов за раз",
        )

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
    content = (
        "{"
        f'"posts":{json.dumps(posts)},'
        f'"selected_post":{json.dumps(selected)},'
        f'"chart":{fig.to_json()}'
        "}"
    )
    return Response(content=content, media_type="application/json")


@app.post("/api/upload/chart/html")
async def chart_html_from_upload(
    files: list[UploadFile] = File(...),
    post_number: str | None = Query(default=None),
    time_start: str | None = Query(default=None),
    time_end: str | None = Query(default=None),
    freq_min: float | None = Query(default=None),
    freq_max: float | None = Query(default=None),
    show_values: bool = Query(default=True),
    hide_zero_rows: bool = Query(default=False),
    hide_zero_columns: bool = Query(default=False),
    time_bin_minutes: int = Query(default=TIME_BIN_MINUTES, ge=1),
    freq_bin_mhz: int = Query(default=FREQ_BIN_MHZ, ge=1),
) -> Response:
    if not files:
        raise HTTPException(status_code=400, detail="Загрузите хотя бы один CSV файл")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Можно загрузить не больше {MAX_UPLOAD_FILES} файлов за раз",
        )

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
    return _html_download_response(
        fig,
        selected,
        show_values=show_values,
        hide_zero_rows=hide_zero_rows,
        hide_zero_columns=hide_zero_columns,
    )
