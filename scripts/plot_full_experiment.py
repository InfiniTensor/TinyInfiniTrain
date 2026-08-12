#!/usr/bin/env python3

import argparse
import csv
import html
from pathlib import Path


def read_rows(path: Path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def moving_average(values, window):
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    return [
        (prefix[index + 1] - prefix[max(0, index + 1 - window)])
        / min(window, index + 1)
        for index in range(len(values))
    ]


def scale(value, lower, upper, start, end):
    if upper == lower:
        return (start + end) / 2
    return start + (value - lower) * (end - start) / (upper - lower)


def panel(
    svg,
    x,
    y,
    width,
    height,
    title,
    x_label,
    y_label,
    x_values,
    series,
    x_ticks,
    y_ticks,
):
    left, right, top, bottom = 58, 18, 34, 46
    plot_x, plot_y = x + left, y + top
    plot_w, plot_h = width - left - right, height - top - bottom
    all_y = [value for _, values, _ in series for value in values]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(all_y), max(all_y)
    y_padding = max(0.05, (y_max - y_min) * 0.12)
    y_min, y_max = y_min - y_padding, y_max + y_padding

    svg.append(
        f'<text x="{x + 4}" y="{y + 18}" class="title">{html.escape(title)}</text>'
    )
    svg.append(
        f'<rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" class="frame"/>'
    )
    for tick in y_ticks:
        py = scale(tick, y_min, y_max, plot_y + plot_h, plot_y)
        svg.append(
            f'<line x1="{plot_x}" y1="{py:.2f}" x2="{plot_x + plot_w}" y2="{py:.2f}" class="grid"/>'
        )
        svg.append(
            f'<text x="{plot_x - 8}" y="{py + 4:.2f}" text-anchor="end">{tick:g}</text>'
        )
    for tick in x_ticks:
        px = scale(tick, x_min, x_max, plot_x, plot_x + plot_w)
        svg.append(
            f'<text x="{px:.2f}" y="{plot_y + plot_h + 20}" text-anchor="middle">{tick:g}</text>'
        )
    svg.append(
        f'<text x="{plot_x + plot_w / 2}" y="{y + height - 5}" text-anchor="middle">{html.escape(x_label)}</text>'
    )
    svg.append(
        f'<text x="{x + 13}" y="{plot_y + plot_h / 2}" text-anchor="middle" '
        f'transform="rotate(-90 {x + 13} {plot_y + plot_h / 2})">{html.escape(y_label)}</text>'
    )

    for label, values, color in series:
        points = []
        for xv, yv in zip(x_values, values):
            px = scale(xv, x_min, x_max, plot_x, plot_x + plot_w)
            py = scale(yv, y_min, y_max, plot_y + plot_h, plot_y)
            points.append(f"{px:.2f},{py:.2f}")
        svg.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        if len(values) <= 10:
            for point in points:
                px, py = point.split(",")
                svg.append(f'<circle cx="{px}" cy="{py}" r="3.5" fill="{color}"/>')
        legend_x = (
            plot_x + 8 + 100 * sum(1 for existing, _, _ in series if existing < label)
        )
        svg.append(
            f'<line x1="{legend_x}" y1="{plot_y + 12}" x2="{legend_x + 20}" y2="{plot_y + 12}" stroke="{color}" stroke-width="3"/>'
        )
        svg.append(
            f'<text x="{legend_x + 26}" y="{plot_y + 16}">{html.escape(label)}</text>'
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plot TinyInfiniTrain full-data experiment metrics"
    )
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    steps = read_rows(args.experiment_dir / "steps.csv")
    epochs = read_rows(args.experiment_dir / "epochs.csv")
    global_steps = [int(row["global_step"]) for row in steps]
    step_losses = [float(row["loss"]) for row in steps]
    smoothed_losses = moving_average(step_losses, 200)
    sampled_indices = list(range(0, len(global_steps), 10))
    if sampled_indices[-1] != len(global_steps) - 1:
        sampled_indices.append(len(global_steps) - 1)
    sampled_steps = [global_steps[index] for index in sampled_indices]
    sampled_losses = [smoothed_losses[index] for index in sampled_indices]
    completed_epochs = [int(row["epoch"]) + 1 for row in epochs]
    train_losses = [float(row["train_mean_loss"]) for row in epochs]
    validation_losses = [float(row["val_mean_loss"]) for row in epochs]
    perplexities = [float(row["val_perplexity"]) for row in epochs]

    width, height = 1320, 410
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font:12px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#172033}.title{font-size:15px;font-weight:650}.heading{font-size:18px;font-weight:700}.frame{fill:#fff;stroke:#94a3b8}.grid{stroke:#dbe3ee;stroke-width:1}</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="660" y="24" text-anchor="middle" class="heading">TinyInfiniTrain GPT-2 124M — A100 full-data run</text>',
    ]
    panel(
        svg,
        10,
        36,
        425,
        360,
        "Step loss (200-step moving mean)",
        "Global optimizer step",
        "Cross-entropy loss",
        sampled_steps,
        [("Train", sampled_losses, "#2563eb")],
        [1, 2500, 5000, 7500, 9540],
        [1, 2, 3, 4],
    )
    panel(
        svg,
        447,
        36,
        425,
        360,
        "Full-epoch mean loss",
        "Completed epoch",
        "Cross-entropy loss",
        completed_epochs,
        [
            ("Train", train_losses, "#2563eb"),
            ("Validation", validation_losses, "#dc2626"),
        ],
        completed_epochs,
        [1, 2, 3, 4, 5, 6],
    )
    panel(
        svg,
        884,
        36,
        425,
        360,
        "Validation perplexity",
        "Completed epoch",
        "Perplexity",
        completed_epochs,
        [("PPL", perplexities, "#7c3aed")],
        completed_epochs,
        [50, 100, 150, 200, 250],
    )
    best_x = scale(
        completed_epochs[0],
        min(completed_epochs),
        max(completed_epochs),
        447 + 58,
        447 + 425 - 18,
    )
    best_y = scale(
        validation_losses[0],
        min(train_losses + validation_losses) - 0.5,
        max(train_losses + validation_losses) + 0.5,
        36 + 34 + 360 - 34 - 46,
        36 + 34,
    )
    svg.append(
        f'<text x="{best_x + 9:.2f}" y="{best_y - 8:.2f}" fill="#166534">best val 3.925</text>'
    )
    svg.append("</svg>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(svg) + "\n")


if __name__ == "__main__":
    main()
