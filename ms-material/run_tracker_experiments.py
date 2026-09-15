import copy
import time

import cv2
import numpy as np

from ms_tracker import MeanShiftTracker, MSParams
from sequence_utils import VOTSequence


DATASET_PATH = r"c:\Users\dino8\Desktop\Sola\MAG\1_letnik\NMRV\Assignment2\vot2014"
REQUIRED_SEQUENCES = ["polarbear", "hand1", "fernando", "drunk", "basketball"]


def evaluate_sequence(dataset_path, sequence_name, params):
    # Oceni sledilnik na eni sekvenci po enakem protokolu kot run_tracker.py.
    sequence = VOTSequence(dataset_path, sequence_name)
    tracker = MeanShiftTracker(params)
    init_frame = 0
    n_failures = 0
    frame_idx = 0
    time_all = 0.0

    # Zanka čez frame-e; ob odpovedi preskočimo 5 frame-ov in reinicializiramo.
    while frame_idx < sequence.length():
        img = cv2.imread(sequence.frame(frame_idx))
        if frame_idx == init_frame:
            # Inicializacija na GT pravokotniku.
            t0 = time.time()
            tracker.initialize(img, sequence.get_annotation(
                frame_idx, type="rectangle"))
            time_all += time.time() - t0
            predicted_bbox = sequence.get_annotation(
                frame_idx, type="rectangle")
        else:
            # Običajno sledenje na naslednjem frame-u.
            t0 = time.time()
            predicted_bbox = tracker.track(img)
            time_all += time.time() - t0

        # Prekrivanje služi kot kriterij odpovedi.
        gt_bb = sequence.get_annotation(frame_idx, type="rectangle")
        o = sequence.overlap(predicted_bbox, gt_bb)

        if o > 0:
            frame_idx += 1
        else:
            # VOT-like: ob odpovedi preskok + ponovna inicializacija.
            frame_idx += 5
            init_frame = frame_idx
            n_failures += 1

    # FPS je povprečje na celotni sekvenci (vključno z init koraki).
    fps = sequence.length() / max(time_all, 1e-8)
    return {"sequence": sequence_name, "fps": fps, "failures": n_failures}


def evaluate_sequences(dataset_path, sequence_names, params):
    # Oceni isto nastavitev na več sekvencah.
    # copy.deepcopy zagotovi, da vsaka sekvenca začne s čistimi parametri.
    rows = []
    for seq in sequence_names:
        rows.append(evaluate_sequence(
            dataset_path, seq, copy.deepcopy(params)))
    return rows


def print_table(title, rows):
    # Lep izpis tabele in agregatov za terminal/report.
    print(f"\n=== {title} ===")
    print(f"{'Sequence':<14} {'FPS':>8} {'Failures':>10}")
    print("-" * 34)
    for r in rows:
        print(f"{r['sequence']:<14} {r['fps']:>8.1f} {r['failures']:>10d}")
    total_f = int(sum(r["failures"] for r in rows))
    avg_fps = float(np.mean([r["fps"] for r in rows])) if rows else 0.0
    print("-" * 34)
    print(f"{'TOTAL/AVG':<14} {avg_fps:>8.1f} {total_f:>10d}")


def run_required():
    # Obvezni del: vsaj 5 sekvenc in poročanje o odpovedih.
    params = MSParams()
    rows = evaluate_sequences(DATASET_PATH, REQUIRED_SEQUENCES, params)
    print_table("Required: 5-sequence result", rows)
    return rows


def run_param_sweep():
    # Razširjeni del: vpliv parametrov na robustnost in hitrost.
    print("\n=== Extra: Parameter impact on failures (5 sequences) ===")
    print("Format: setting -> total_failures, avg_fps")

    # Pripravimo seznam nastavitev za sistematičen preizkus.
    settings = []

    # Število binov histograma.
    for nb in [8, 16, 32]:
        p = MSParams()
        p.nbins = nb
        settings.append((f"nbins={nb}", p))

    # Velikost iskalnega okna.
    for ws in [1.5, 2.0, 2.5]:
        p = MSParams()
        p.enlarge_factor = ws
        settings.append((f"window_scale={ws}", p))

    # "Velikost" jedra pri modeliranju objekta.
    for hs in [0.8, 1.0, 1.2]:
        p = MSParams()
        p.hist_kernel_sigma = hs
        settings.append((f"kernel_sigma={hs}", p))

    # Ustavitveni prag.
    for eps in [1.0, 0.5, 0.2]:
        p = MSParams()
        p.stop_eps = eps
        settings.append((f"stop_eps={eps}", p))

    # Maksimalno število iteracij.
    for iters in [10, 20, 35]:
        p = MSParams()
        p.max_iters = iters
        settings.append((f"max_iters={iters}", p))

    # Hitrost posodabljanja modela tarče.
    for alpha in [0.0, 0.05, 0.2]:
        p = MSParams()
        p.model_alpha = alpha
        settings.append((f"model_alpha={alpha}", p))

    # Zagon in izpis agregatov za vsako nastavitev.
    for name, params in settings:
        rows = evaluate_sequences(DATASET_PATH, REQUIRED_SEQUENCES, params)
        total_f = int(sum(r["failures"] for r in rows))
        avg_fps = float(np.mean([r["fps"] for r in rows]))
        print(f"{name:<18} -> failures={total_f:>3d}, avg_fps={avg_fps:>6.1f}")


def run_bonus():
    # Bonus 1: primerjava barvnih prostorov.
    print("\n=== Bonus: Color-space comparison (5 sequences) ===")
    for cs in ["BGR", "HSV", "Lab", "YCrCb"]:
        p = MSParams()
        p.color_space = cs
        rows = evaluate_sequences(DATASET_PATH, REQUIRED_SEQUENCES, p)
        total_f = int(sum(r["failures"] for r in rows))
        avg_fps = float(np.mean([r["fps"] for r in rows]))
        print(f"{cs:<6} -> failures={total_f:>3d}, avg_fps={avg_fps:>6.1f}")

    # Bonus 2: ablacijski test modela ozadja.
    print("\n=== Bonus: Background model ablation (5 sequences) ===")
    p0 = MSParams()
    p0.use_background_model = False
    r0 = evaluate_sequences(DATASET_PATH, REQUIRED_SEQUENCES, p0)
    f0 = int(sum(r["failures"] for r in r0))
    fps0 = float(np.mean([r["fps"] for r in r0]))

    p1 = MSParams()
    p1.use_background_model = True
    p1.background_weight = 0.5
    p1.background_scale = 1.8
    r1 = evaluate_sequences(DATASET_PATH, REQUIRED_SEQUENCES, p1)
    f1 = int(sum(r["failures"] for r in r1))
    fps1 = float(np.mean([r["fps"] for r in r1]))

    print(f"no_bg_model  -> failures={f0:>3d}, avg_fps={fps0:>6.1f}")
    print(f"with_bg_model-> failures={f1:>3d}, avg_fps={fps1:>6.1f}")


def run_failure_cases(required_rows):
    # Iz obveznih rezultatov izluščimo tipične težke in lahke primere.
    print("\n=== Extra: Failure cases ===")
    worst = sorted(
        required_rows, key=lambda x: x["failures"], reverse=True)[:2]
    best = sorted(required_rows, key=lambda x: x["failures"])[:2]
    print("Most difficult sequences in this run:")
    for r in worst:
        print(f"  {r['sequence']}: failures={r['failures']}, fps={r['fps']:.1f}")
    print("Most stable sequences in this run:")
    for r in best:
        print(f"  {r['sequence']}: failures={r['failures']}, fps={r['fps']:.1f}")


def main():
    # Glavni potek: obvezni del -> analiza napak -> sweep parametrov -> bonus.
    req_rows = run_required()
    run_failure_cases(req_rows)
    run_param_sweep()
    run_bonus()


if __name__ == "__main__":
    # Neposreden zagon skripte iz terminala.
    main()
