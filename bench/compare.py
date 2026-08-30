from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from bench.analyze import read_jsonl, summarize


def comparison_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    configurations: set[tuple[str, int, str]] = set()
    for path in paths:
        records = read_jsonl([path])
        loadgen = next(
            (record for record in records if record.get("type") == "loadgen_config"),
            None,
        )
        server = next(
            (record for record in records if record.get("type") == "server_metrics"),
            None,
        )
        if loadgen is None or server is None:
            raise ValueError(f"{path} is missing load-generator or server configuration")
        mode = server.get("provenance", {}).get("config", {}).get("mode")
        if mode not in {"sequential", "static", "continuous"}:
            raise ValueError(f"{path} has an invalid or missing scheduling mode")
        concurrency = int(loadgen["concurrency"])
        workload = str(loadgen["length_workload"])
        configuration = (mode, concurrency, workload)
        if configuration in configurations:
            raise ValueError(f"Duplicate comparison configuration {configuration}")
        configurations.add(configuration)
        rows.append(
            {
                "mode": mode,
                "concurrency": concurrency,
                "workload": workload,
                **summarize(records),
                "source": str(path),
            }
        )
    return sorted(
        rows,
        key=lambda row: (row["workload"], row["mode"], row["concurrency"]),
    )


def write_comparison(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        raise ValueError("At least one comparison row is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    workloads = sorted({row["workload"] for row in rows})
    modes = ("sequential", "static", "continuous")
    figure, axes = plt.subplots(
        len(workloads),
        2,
        figsize=(11, 4 * len(workloads)),
        squeeze=False,
    )
    for row_index, workload in enumerate(workloads):
        workload_rows = [row for row in rows if row["workload"] == workload]
        for mode in modes:
            mode_rows = sorted(
                (row for row in workload_rows if row["mode"] == mode),
                key=lambda row: row["concurrency"],
            )
            if not mode_rows:
                continue
            concurrency = [row["concurrency"] for row in mode_rows]
            axes[row_index][0].plot(
                concurrency,
                [row["throughput_tokens_per_second"] for row in mode_rows],
                marker="o",
                label=mode,
            )
            axes[row_index][1].plot(
                concurrency,
                [row["end_to_end_ms_p99"] for row in mode_rows],
                marker="o",
                label=mode,
            )
        axes[row_index][0].set(
            title=f"{workload}: throughput",
            xlabel="Concurrency",
            ylabel="Output tokens/second",
        )
        axes[row_index][1].set(
            title=f"{workload}: p99 end-to-end latency",
            xlabel="Concurrency",
            ylabel="Latency (ms)",
        )
        for axis in axes[row_index]:
            axis.set_xticks(
                sorted({row["concurrency"] for row in workload_rows})
            )
            axis.grid(alpha=0.25)
            axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "mode_comparison.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze batching-mode comparisons")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = comparison_rows(args.inputs)
    write_comparison(rows, args.output_dir)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
