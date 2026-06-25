from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go

from parser import ParsedCsv

TIME_BIN_MINUTES = 10
FREQ_BIN_MHZ = 10
CELL_WIDTH_PX = 56
CELL_HEIGHT_PX = 28
GRID_GAP_PX = 2
MIN_CHART_WIDTH_PX = 800
MIN_CHART_HEIGHT_PX = 400
CHART_MARGIN = {"l": 80, "r": 80, "t": 90, "b": 120}


def _combine_post_data(files: list[ParsedCsv]) -> pd.DataFrame:
    frames = [f.data.assign(source=f.source) for f in files]
    return pd.concat(frames, ignore_index=True)


def _floor_freq(freq: float) -> float:
    return (freq // FREQ_BIN_MHZ) * FREQ_BIN_MHZ


def _build_time_bins(start: datetime, end: datetime) -> pd.DatetimeIndex:
    start_bin = pd.Timestamp(start).floor(f"{TIME_BIN_MINUTES}min")
    end_bin = pd.Timestamp(end).floor(f"{TIME_BIN_MINUTES}min")
    if start_bin > end_bin:
        raise ValueError("Начало временного диапазона должно быть раньше конца.")
    return pd.date_range(start=start_bin, end=end_bin, freq=f"{TIME_BIN_MINUTES}min")


def _build_freq_bins(freq_min: float, freq_max: float) -> list[float]:
    min_bin = _floor_freq(freq_min)
    max_bin = _floor_freq(freq_max)
    if min_bin > max_bin:
        raise ValueError("Минимальная частота должна быть меньше или равна максимальной.")
    return list(range(int(max_bin), int(min_bin) - FREQ_BIN_MHZ, -FREQ_BIN_MHZ))


def build_heatmap_figure(
    files: list[ParsedCsv],
    post_number: str,
    *,
    time_start: datetime | None = None,
    time_end: datetime | None = None,
    freq_min: float | None = None,
    freq_max: float | None = None,
    show_values: bool = True,
) -> go.Figure:
    df = _combine_post_data(files)

    if time_start is not None:
        df = df[df["time"] >= pd.Timestamp(time_start)]
    if time_end is not None:
        df = df[df["time"] <= pd.Timestamp(time_end)]
    if freq_min is not None:
        df = df[df["frequency"] >= freq_min]
    if freq_max is not None:
        df = df[df["frequency"] <= freq_max]

    df["time_bin"] = df["time"].dt.floor(f"{TIME_BIN_MINUTES}min")
    df["freq_bin"] = (df["frequency"] // FREQ_BIN_MHZ) * FREQ_BIN_MHZ

    if df.empty:
        pivot = pd.DataFrame()
    else:
        pivot = (
            df.groupby(["freq_bin", "time_bin"])
            .size()
            .reset_index(name="count")
            .pivot(index="freq_bin", columns="time_bin", values="count")
            .fillna(0)
            .astype(int)
        )

    use_fixed_time = time_start is not None and time_end is not None
    use_fixed_freq = freq_min is not None and freq_max is not None

    if use_fixed_time:
        time_bins = _build_time_bins(time_start, time_end)
    elif not pivot.empty:
        time_bins = pivot.columns
    else:
        time_bins = pd.DatetimeIndex([])

    if use_fixed_freq:
        freq_bins = _build_freq_bins(freq_min, freq_max)
    elif not pivot.empty:
        freq_bins = sorted(pivot.index.unique(), reverse=True)
    else:
        freq_bins = []

    if len(time_bins) and len(freq_bins):
        pivot = pivot.reindex(index=freq_bins, columns=time_bins, fill_value=0).astype(int)
    elif use_fixed_time or use_fixed_freq:
        pivot = pd.DataFrame(0, index=freq_bins, columns=time_bins, dtype=int)

    time_labels = [ts.strftime("%Y-%m-%d %H:%M") for ts in pivot.columns]
    freq_labels = [f"{int(f)} МГц" for f in pivot.index]

    metadata = files[0].metadata
    range_note = ""
    if use_fixed_time or use_fixed_freq:
        parts = []
        if use_fixed_time:
            parts.append(
                f"Время: {time_start:%Y-%m-%d %H:%M} — {time_end:%Y-%m-%d %H:%M}"
            )
        if use_fixed_freq:
            parts.append(f"Частота: {int(freq_min)}—{int(freq_max)} МГц")
        range_note = " | " + " | ".join(parts)

    title = (
        f"Пост {post_number} — распределение частот<br>"
        f"<sup>Частота анализа: {metadata.analysis_frequency} | "
        f"Полоса: {metadata.analysis_band} | "
        f"Начало: {metadata.analysis_start_time}{range_note}</sup>"
    )

    n_cols = len(time_labels)
    n_rows = len(freq_labels)
    chart_width = max(MIN_CHART_WIDTH_PX, n_cols * CELL_WIDTH_PX + 160)
    chart_height = max(MIN_CHART_HEIGHT_PX, n_rows * CELL_HEIGHT_PX + 200)

    heatmap_kwargs: dict = {
        "z": pivot.values,
        "x": time_labels,
        "y": freq_labels,
        "colorscale": "YlOrRd",
        "colorbar": dict(title="Кол-во"),
        "xgap": GRID_GAP_PX,
        "ygap": GRID_GAP_PX,
        "hovertemplate": (
            "Время: %{x}<br>"
            "Частота: %{y}<br>"
            "Количество: %{z}<extra></extra>"
        ),
    }
    if show_values:
        heatmap_kwargs.update(
            text=pivot.values,
            texttemplate="%{text}",
            textfont={"size": 10},
        )
    else:
        heatmap_kwargs["texttemplate"] = ""

    fig = go.Figure(data=go.Heatmap(**heatmap_kwargs))

    fig.update_layout(
        title=title,
        width=chart_width,
        height=chart_height,
        autosize=False,
        dragmode="zoom",
        plot_bgcolor="#ffffff",
        xaxis_title=f"Время (шаг {TIME_BIN_MINUTES} мин)",
        yaxis_title=f"Частота (шаг {FREQ_BIN_MHZ} МГц)",
        xaxis={
            "tickangle": -45,
            "type": "category",
            "fixedrange": False,
            "automargin": True,
            "showgrid": True,
            "gridcolor": "#d0d7de",
            "gridwidth": 1,
            "showline": True,
            "linewidth": 1,
            "linecolor": "#9aa7b8",
        },
        yaxis={
            "type": "category",
            "fixedrange": False,
            "automargin": True,
            "showgrid": True,
            "gridcolor": "#d0d7de",
            "gridwidth": 1,
            "showline": True,
            "linewidth": 1,
            "linecolor": "#9aa7b8",
        },
        margin=CHART_MARGIN,
    )

    return fig
