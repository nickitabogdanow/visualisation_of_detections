from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class CsvMetadata:
    post_number: str
    analysis_frequency: str
    analysis_band: str
    buffer_duration: str
    analysis_start_time: str
    table_name: str


@dataclass
class ParsedCsv:
    metadata: CsvMetadata
    data: pd.DataFrame
    source: str


TIME_COLUMN_CANDIDATES = (
    "время",
    "time",
    "timestamp",
    "дата",
    "datetime",
    "временная метка",
    "метка времени",
)

FREQ_COLUMN_CANDIDATES = (
    "частота",
    "frequency",
    "freq",
    "f, mhz",
    "f_mhz",
    "частота, mhz",
    "частота (mhz)",
)


def _normalize_header(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {_normalize_header(c): c for c in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for col in columns:
        norm = _normalize_header(col)
        if any(candidate in norm for candidate in candidates):
            return col
    return None


def _read_raw_lines(content: str | bytes) -> list[str]:
    if isinstance(content, bytes):
        text = content.decode("utf-8-sig")
    else:
        text = content
    return text.splitlines()


def parse_metadata(lines: list[str]) -> tuple[CsvMetadata, int]:
    if len(lines) < 8:
        raise ValueError("CSV должен содержать минимум 8 строк (метаданные + заголовки).")

    def cell(row_idx: int, col_idx: int = 1) -> str:
        parts = [p.strip() for p in lines[row_idx].split(",")]
        if len(parts) <= col_idx:
            return ""
        return parts[col_idx]

    metadata = CsvMetadata(
        post_number=cell(0),
        analysis_frequency=cell(1),
        analysis_band=cell(2),
        buffer_duration=cell(3),
        analysis_start_time=cell(4),
        table_name=lines[6].strip().strip(","),
    )
    return metadata, 7


_TIME_ONLY_PATTERN = re.compile(r"^\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?$")


def _time_only_mask(values: pd.Series) -> pd.Series:
    as_str = values.astype(str).str.strip()
    valid = values.notna() & (as_str.str.lower() != "nan") & (as_str != "")
    return valid & as_str.str.match(_TIME_ONLY_PATTERN.pattern, na=False)


def _parse_times_with_metadata(
    values: pd.Series, analysis_start_time: str
) -> pd.Series:
    metadata_dt = pd.to_datetime(analysis_start_time, errors="coerce", format="mixed")
    if pd.isna(metadata_dt):
        raise ValueError(
            f"Некорректное время начала анализа в метаданных: {analysis_start_time!r}"
        )

    mask = _time_only_mask(values)
    if not mask.any():
        return pd.to_datetime(values, errors="coerce", format="mixed")

    base_day = metadata_dt.normalize()
    if mask.all():
        time_parts = pd.to_datetime(values.astype(str).str.strip(), errors="coerce", format="mixed")
        return base_day + (time_parts - time_parts.dt.normalize())

    result = pd.to_datetime(values, errors="coerce", format="mixed")
    time_parts = pd.to_datetime(values[mask].astype(str).str.strip(), errors="coerce", format="mixed")
    result.loc[mask] = base_day + (time_parts - time_parts.dt.normalize())
    return result


def parse_csv_content(content: str | bytes, source: str = "upload") -> ParsedCsv:
    lines = _read_raw_lines(content)
    metadata, header_row_idx = parse_metadata(lines)

    table_text = "\n".join(lines[header_row_idx:])
    data = pd.read_csv(io.StringIO(table_text))

    if data.empty:
        raise ValueError(f"Таблица данных пуста ({source}).")

    time_col = _find_column(list(data.columns), TIME_COLUMN_CANDIDATES)
    freq_col = _find_column(list(data.columns), FREQ_COLUMN_CANDIDATES)

    if time_col is None:
        raise ValueError(
            f"Не найден столбец времени. Доступные столбцы: {', '.join(map(str, data.columns))}"
        )
    if freq_col is None:
        raise ValueError(
            f"Не найден столбец частоты. Доступные столбцы: {', '.join(map(str, data.columns))}"
        )

    data = data.rename(columns={time_col: "time", freq_col: "frequency"})
    data["time"] = _parse_times_with_metadata(data["time"], metadata.analysis_start_time)
    data["frequency"] = pd.to_numeric(data["frequency"], errors="coerce")

    data = data.dropna(subset=["time", "frequency"])
    if data.empty:
        raise ValueError(f"После очистки не осталось валидных строк ({source}).")

    return ParsedCsv(metadata=metadata, data=data[["time", "frequency"]], source=source)


def parse_csv_file(path: Path) -> ParsedCsv:
    content = path.read_bytes()
    return parse_csv_content(content, source=path.name)


def load_csv_files(paths: list[Path]) -> dict[str, list[ParsedCsv]]:
    by_post: dict[str, list[ParsedCsv]] = {}
    for path in paths:
        parsed = parse_csv_file(path)
        post = parsed.metadata.post_number or "unknown"
        by_post.setdefault(post, []).append(parsed)
    return by_post
