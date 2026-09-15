import numpy as np
import cv2

from ex2_utils import generate_responses_1
from ms_tracker import mean_shift_mode_seeking


def generate_custom_responses():
    # Ustvarimo lastno testno površino z več vrhovi (custom funkcija).
    # S tem pokrijemo dodatni del naloge za mode-seeking.
    responses = np.zeros((120, 120), dtype=np.float32)
    # Diskretni "impulzi", ki bodo po glajenju postali vrhovi.
    responses[25, 25] = 0.7
    responses[90, 85] = 1.0
    responses[60, 45] = 0.45
    # Glajenje spremeni impulze v zvezno pokrajino odzivov.
    responses = cv2.GaussianBlur(responses, (0, 0), sigmaX=7.0, sigmaY=7.0)
    # Normalizacija za bolj primerljive rezultate med poskusi.
    responses = responses / (responses.max() + 1e-8)
    return responses


def run_on_map(name, responses):
    # Za dano pokrajino izvedemo mrežo eksperimentov.
    print(f"\n=== {name} ===")

    # Začetne točke, velikosti jeder in kriteriji ustavitve.
    starts = [(15, 15), (35, 70), (65, 40), (85, 80)]
    bandwidths = [11, 21, 31]
    criteria = [1.0, 0.5, 0.1]
    kernels = ["epanechnikov", "gaussian"]
    all_steps = []

    # Izčrpen preizkus vseh kombinacij.
    for k in kernels:
        for bw in bandwidths:
            for eps in criteria:
                print(f"\nkernel={k}, bandwidth={bw}, stop_eps={eps}")
                for s in starts:
                    # Zagon mean-shift mode-seeking.
                    mode, n_steps = mean_shift_mode_seeking(
                        responses,
                        start_pos=s,
                        bandwidth=bw,
                        max_iters=100,
                        eps=eps,
                        kernel_type=k,
                    )
                    # Shranimo korake zaradi kasnejšega povzetka.
                    all_steps.append(n_steps)
                    print(
                        f"  start={s} -> mode=({mode[0]:.2f}, {mode[1]:.2f}), steps={n_steps}"
                    )
    # Kratek povzetek hitrosti konvergence za trenutno pokrajino.
    all_steps = np.array(all_steps, dtype=np.float32)
    print("\nsummary:")
    print(f"  avg steps: {float(all_steps.mean()):.2f}")
    print(f"  max steps: {int(all_steps.max())}")
    print(f"  min steps: {int(all_steps.min())}")


def run():
    # Glavna točka: najprej uradna funkcija, nato lastna funkcija.
    print("Mean-shift mode-seeking experiments")
    run_on_map("Provided function (generate_responses_1)", generate_responses_1().astype(np.float32))
    run_on_map("Custom function (three peaks)", generate_custom_responses().astype(np.float32))


if __name__ == "__main__":
    # Omogoča neposreden zagon skripte iz terminala.
    run()
