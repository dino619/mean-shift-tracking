import argparse
from pathlib import Path

import cv2
import numpy as np

from ex2_utils import generate_responses_1
from run_mode_seeking import generate_custom_responses
from ms_tracker import mean_shift_mode_seeking


def _as_heatmap(responses):
    r = responses.astype(np.float32)
    r = (r - r.min()) / (r.max() - r.min() + 1e-8)
    g = (r * 255.0).astype(np.uint8)
    return cv2.applyColorMap(g, cv2.COLORMAP_VIRIDIS)


def _draw_legend(canvas, lines, x, y, scale):
    font_scale = 0.45 * max(1.0, scale / 4.0)
    thickness = max(1, scale // 6)
    box_w = max(16, int(14 * scale / 4))
    box_h = max(10, int(12 * scale / 4))
    line_gap = max(18, int(20 * scale / 4))

    for txt, color in lines:
        cv2.rectangle(canvas, (x, y - box_h), (x + box_w, y), color, thickness=-1)
        cv2.putText(
            canvas,
            txt,
            (x + box_w + 8, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_gap


def plot_paths(
    responses,
    starts,
    kernel_type="epanechnikov",
    bandwidth=21,
    eps=0.5,
    max_iters=100,
    title="Mean-shift trajectories",
    out_path="mode_seeking_paths.png",
    scale=8,
    style="clean",
):
    scale = int(max(1, scale))
    heat = _as_heatmap(responses)
    heat = cv2.resize(
        heat,
        (heat.shape[1] * scale, heat.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = heat.copy()

    palette = [
        (255, 0, 0),
        (0, 255, 255),
        (0, 255, 0),
        (255, 255, 0),
        (255, 0, 255),
        (0, 165, 255),
    ]
    legend = []

    for i, s in enumerate(starts):
        mode, n_steps, path = mean_shift_mode_seeking(
            responses,
            start_pos=s,
            bandwidth=bandwidth,
            max_iters=max_iters,
            eps=eps,
            kernel_type=kernel_type,
            return_path=True,
        )
        color = palette[i % len(palette)]
        pts = np.array(path, dtype=np.int32)

        # Pot mean-shift centra.
        for j in range(1, len(pts)):
            p0 = (int(pts[j - 1, 0] * scale), int(pts[j - 1, 1] * scale))
            p1 = (int(pts[j, 0] * scale), int(pts[j, 1] * scale))
            cv2.line(canvas, p0, p1, color, 2 + scale // 4, cv2.LINE_AA)

        # Začetek (X) in konec (krog).
        sx, sy = int(pts[0, 0] * scale), int(pts[0, 1] * scale)
        ex, ey = int(pts[-1, 0] * scale), int(pts[-1, 1] * scale)
        mark = max(4, scale // 2)
        # V "clean" načinu je začetek vedno rdeč križec (podobno kot v vzorčnem poročilu).
        start_color = (0, 0, 255) if style == "clean" else color
        cv2.line(canvas, (sx - mark, sy - mark), (sx + mark, sy + mark), start_color, 2 + scale // 5, cv2.LINE_AA)
        cv2.line(canvas, (sx - mark, sy + mark), (sx + mark, sy - mark), start_color, 2 + scale // 5, cv2.LINE_AA)
        cv2.circle(canvas, (ex, ey), max(4, scale // 2), color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (ex, ey), max(6, scale // 2 + 2), (255, 255, 255), 1 + scale // 8, cv2.LINE_AA)

        legend.append(
            (
                f"start={s} end=({mode[0]:.1f},{mode[1]:.1f}) steps={n_steps}",
                color,
            )
        )

    # "clean" stil: brez overlay teksta, primeren za poročilo.
    if style != "clean":
        top_h = max(32, int(30 * scale / 4))
        legend_h = max(110, int((22 * len(legend) + 10) * scale / 4))

        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], top_h), (0, 0, 0), -1)
        cv2.rectangle(canvas, (0, top_h), (canvas.shape[1], top_h + legend_h), (0, 0, 0), -1)

        cv2.putText(
            canvas,
            title,
            (8, int(top_h * 0.72)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45 * max(1.0, scale / 4.0),
            (255, 255, 255),
            1 + scale // 8,
            cv2.LINE_AA,
        )
        _draw_legend(
            canvas,
            legend,
            x=10,
            y=top_h + max(18, int(20 * scale / 4)),
            scale=scale,
        )

    out_path = Path(out_path)
    cv2.imwrite(str(out_path), canvas)
    print(f"Saved: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Draw mean-shift trajectories on response map.")
    parser.add_argument("--map", choices=["provided", "custom"], default="provided")
    parser.add_argument("--kernel", choices=["epanechnikov", "gaussian"], default="epanechnikov")
    parser.add_argument("--bandwidth", type=int, default=21)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--out", type=str, default="mode_seeking_paths.png")
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--style", choices=["clean", "full"], default="clean")
    args = parser.parse_args()

    if args.map == "provided":
        responses = generate_responses_1().astype(np.float32)
        title_map = "provided"
    else:
        responses = generate_custom_responses().astype(np.float32)
        title_map = "custom"

    starts = [(15, 15), (35, 70), (65, 40), (85, 80)]
    title = f"map={title_map} kernel={args.kernel} bw={args.bandwidth} eps={args.eps}"
    plot_paths(
        responses=responses,
        starts=starts,
        kernel_type=args.kernel,
        bandwidth=args.bandwidth,
        eps=args.eps,
        title=title,
        out_path=args.out,
        scale=args.scale,
        style=args.style,
    )


if __name__ == "__main__":
    main()
