from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path


POSTS = ("101", "202", "303")
DEFAULT_FILE_COUNT = 4000
ROWS_PER_FILE = 24
START_TIME = datetime(2025, 6, 25, 8, 0, 0)


def build_rows(post: str, file_index: int, rows_per_file: int) -> list[str]:
    rng = random.Random(f"{post}-{file_index}")
    base_time = START_TIME + timedelta(minutes=10 * file_index)
    base_frequency = 2400 + POSTS.index(post) * 100

    rows = []
    for row_index in range(rows_per_file):
        timestamp = base_time + timedelta(minutes=5 * row_index)
        frequency = base_frequency + rng.randrange(0, 101, 5)
        power = rng.randint(-75, -35)
        rows.append(f"{timestamp:%Y-%m-%d %H:%M:%S},{frequency},{power}")

    return rows


def build_file_content(post: str, file_index: int, rows_per_file: int) -> str:
    rows = build_rows(post, file_index, rows_per_file)
    analysis_start = START_TIME + timedelta(minutes=10 * file_index)

    return "\n".join(
        [
            f"номер поста,{post}",
            f"частота анализа,{2400 + POSTS.index(post) * 100} МГц",
            "полоса анализа,20 МГц",
            "длительность буфера анализа,3600",
            f"время начала анализа,{analysis_start:%Y-%m-%d %H:%M:%S}",
            "",
            f"Результаты анализа поста {post}",
            "Время,Частота,Мощность",
            *rows,
            "",
        ]
    )


def distribute_files(total_files: int) -> dict[str, int]:
    per_post = total_files // len(POSTS)
    remainder = total_files % len(POSTS)
    return {
        post: per_post + (1 if index < remainder else 0)
        for index, post in enumerate(POSTS)
    }


def generate_files(output_dir: Path, total_files: int, rows_per_file: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    distribution = distribute_files(total_files)

    for post, file_count in distribution.items():
        for file_index in range(file_count):
            filename = f"generated_post_{post}_{file_index + 1:04d}.csv"
            path = output_dir / filename
            path.write_text(
                build_file_content(post, file_index, rows_per_file),
                encoding="utf-8",
            )

    summary = ", ".join(
        f"post {post}: {file_count}" for post, file_count in distribution.items()
    )
    print(f"Generated {total_files} files in {output_dir}")
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic CSV files for IDF visualisation testing."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory where generated CSV files will be written.",
    )
    parser.add_argument(
        "--files",
        type=int,
        default=DEFAULT_FILE_COUNT,
        help="Total number of files to generate.",
    )
    parser.add_argument(
        "--rows-per-file",
        type=int,
        default=ROWS_PER_FILE,
        help="Number of data rows in each generated CSV file.",
    )
    args = parser.parse_args()

    if args.files <= 0:
        raise SystemExit("--files must be greater than 0")
    if args.rows_per_file <= 0:
        raise SystemExit("--rows-per-file must be greater than 0")

    generate_files(args.output_dir, args.files, args.rows_per_file)


if __name__ == "__main__":
    main()
