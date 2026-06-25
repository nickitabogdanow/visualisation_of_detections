from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from parser import ParsedCsv

TIME_BIN_MINUTES = 10
FREQ_BIN_MHZ = 10


def _combine_post_data(files: list[ParsedCsv]) -> pd.DataFrame:
    frames = [f.data.assign(source=f.source) for f in files]
    return pd.concat(frames, ignore_index=True)


def build_heatmap_figure(files: list[ParsedCsv], post_number: str) -> go.Figure:
    df = _combine_post_data(files)

    df["time_bin"] = df["time"].dt.floor(f"{TIME_BIN_MINUTES}min")
    df["freq_bin"] = (df["frequency"] // FREQ_BIN_MHZ) * FREQ_BIN_MHZ

    pivot = (
        df.groupby(["freq_bin", "time_bin"])
        .size()
        .reset_index(name="count")
        .pivot(index="freq_bin", columns="time_bin", values="count")
        .fillna(0)
        .astype(int)
    )

    pivot = pivot.sort_index(ascending=False)

    time_labels = [ts.strftime("%Y-%m-%d %H:%M") for ts in pivot.columns]
    freq_labels = [f"{int(f)} МГц" for f in pivot.index]

    metadata = files[0].metadata
    title = (
        f"Пост {post_number} — распределение частот<br>"
        f"<sup>Частота анализа: {metadata.analysis_frequency} | "
        f"Полоса: {metadata.analysis_band} | "
        f"Начало: {metadata.analysis_start_time}</sup>"
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=time_labels,
            y=freq_labels,
            colorscale="YlOrRd",
            colorbar=dict(title="Кол-во"),
            hovertemplate=(
                "Время: %{x}<br>"
                "Частота: %{y}<br>"
                "Количество: %{z}<extra></extra>"
            ),
            text=pivot.values,
            texttemplate="%{text}",
            textfont={"size": 10},
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=f"Время (шаг {TIME_BIN_MINUTES} мин)",
        yaxis_title=f"Частота (шаг {FREQ_BIN_MHZ} МГц)",
        xaxis={"tickangle": -45, "type": "category"},
        yaxis={"type": "category"},
        margin={"l": 80, "r": 40, "t": 90, "b": 120},
        height=max(400, len(freq_labels) * 28 + 180),
    )

    return fig
