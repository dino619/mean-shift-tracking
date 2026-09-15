import cv2
import numpy as np

from ex2_utils import (
    Tracker,
    backproject_histogram,
    extract_histogram,
    get_patch,
)


def _kernel_weights(dist2, kernel_type):
    # Izračun uteži mean-shift jedra glede na izbrani tip.
    # dist2 je normalizirana kvadratna razdalja od centra okna.
    if kernel_type == "gaussian":
        # Gaussovo jedro da večjo težo točkam bližje centru.
        return np.exp(-0.5 * dist2).astype(np.float32)
    # Odvod Epanechnikovega jedra je konstantno jedro (uniformno znotraj podpore).
    return (dist2 <= 1.0).astype(np.float32)


def _epanechnikov_kernel_exact(width, height, sigma):
    # Ustvari Epanechnikovo jedro točno enake velikosti kot obliž (patch).
    # To prepreči napake pri množenju mask in jeder zaradi neujemanja dimenzij.
    width = int(max(1, width))
    height = int(max(1, height))
    sigma = float(max(1e-6, sigma))

    # Koordinatna mreža v območju [-1, 1] po obeh oseh.
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xs, ys = np.meshgrid(x, y)

    # Formula Epanechnikovega profila; negativne vrednosti odrežemo na 0.
    kernel = 1.0 - ((xs / sigma) ** 2 + (ys / sigma) ** 2)
    kernel[kernel < 0.0] = 0.0

    # Normalizacija na [0, 1] za stabilne uteži.
    m = float(np.max(kernel))
    if m > 0:
        kernel /= m
    return kernel.astype(np.float32)


def mean_shift_mode_seeking(
    responses,
    start_pos,
    bandwidth,
    max_iters=50,
    eps=1e-3,
    kernel_type="epanechnikov",
    return_path=False,
):
    # responses: 2D površina odzivov
    # start_pos: začetna točka (x, y)
    # bandwidth: velikost iskalnega okna
    # max_iters/eps: ustavitveni pogoji
    h, w = responses.shape[:2]
    cy, cx = float(start_pos[1]), float(start_pos[0])
    bw = int(max(3, round(bandwidth)))
    half = bw // 2

    # Lokalna koordinatna mreža okna, centrirana okoli (0, 0).
    ys, xs = np.mgrid[-half : half + 1, -half : half + 1].astype(np.float32)
    if half > 0:
        # Normalizirana kvadratna razdalja od centra.
        dist2 = (xs / float(half)) ** 2 + (ys / float(half)) ** 2
    else:
        dist2 = np.zeros_like(xs, dtype=np.float32)
    k_weights = _kernel_weights(dist2, kernel_type)

    # Iterativni mean-shift premik centra.
    n_steps = 0
    # Pot centra po iteracijah (uporabno za vizualizacijo konvergence).
    path = [(cx, cy)]
    for it in range(max_iters):
        # Izrežemo lokalni obliž okoli trenutnega centra.
        patch, _ = get_patch(responses, (cx, cy), (bw, bw))
        patch = patch.astype(np.float32)
        if patch.ndim == 3:
            # Za vsak slučaj vzamemo en kanal, če vhod ni strogo 2D.
            patch = patch[:, :, 0]

        # Uteži so produkt lokalnega odziva in jedra.
        weights = patch * k_weights
        s = float(np.sum(weights))
        if s <= 1e-12:
            # Če ni mase, ni informacije za premik.
            break

        # Težiščni premik znotraj okna.
        dx = float(np.sum(weights * xs) / s)
        dy = float(np.sum(weights * ys) / s)

        # Nov center omejimo z mejami slike.
        cx_new = float(np.clip(cx + dx, 0, w - 1))
        cy_new = float(np.clip(cy + dy, 0, h - 1))
        n_steps = it + 1

        # Konvergenca: premik je manjši od praga.
        if (dx * dx + dy * dy) ** 0.5 < eps:
            cx, cy = cx_new, cy_new
            path.append((cx, cy))
            break
        cx, cy = cx_new, cy_new
        path.append((cx, cy))

    # Vrne končni položaj in število iteracij.
    if return_path:
        return (cx, cy), n_steps, path
    return (cx, cy), n_steps


class MSParams:
    def __init__(self):
        # Število binov 3D barvnega histograma.
        self.nbins = 16
        # Faktor velikosti iskalnega okna glede na bbox objekta.
        self.enlarge_factor = 2.0
        # Največ mean-shift iteracij na frame.
        self.max_iters = 20
        # Prag konvergence (v piksljih).
        self.stop_eps = 0.5
        # Hitrost posodobitve modela objekta.
        self.model_alpha = 0.05
        # Sigma za Epanechnikovo jedro pri ekstrakciji histograma.
        self.hist_kernel_sigma = 1.0
        # Tip mean-shift jedra.
        self.ms_kernel_type = "epanechnikov"
        self.color_space = "BGR"  # Bonus: podpira BGR, HSV, Lab, YCrCb
        # Bonus: modeliranje ozadja.
        self.use_background_model = False
        # Teža ozadja v imenovalcu uteži.
        self.background_weight = 0.5
        # Velikost ozadnega obroča glede na bbox objekta.
        self.background_scale = 1.8


class MeanShiftTracker(Tracker):
    def _to_color_space(self, image):
        # Pretvorba v izbran barvni prostor.
        cs = self.parameters.color_space.lower()
        if cs == "hsv":
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if cs == "lab":
            return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        if cs in ("ycrcb", "ycbcr"):
            return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        # Privzeto: BGR (brez pretvorbe).
        return image

    def _normalized_hist(self, patch, weights):
        # Izračun uteženega histograma in L1 normalizacija.
        h = extract_histogram(patch, self.parameters.nbins, weights=weights).astype(np.float32)
        s = float(np.sum(h))
        if s > 0:
            h /= s
        return h

    def initialize(self, image, region):
        # Če je regija podana kot poligon (8 vrednosti), jo pretvorimo v pravokotnik.
        if len(region) == 8:
            x_ = np.array(region[::2], dtype=np.float32)
            y_ = np.array(region[1::2], dtype=np.float32)
            region = [float(np.min(x_)), float(np.min(y_)), float(np.max(x_) - np.min(x_) + 1), float(np.max(y_) - np.min(y_) + 1)]

        # Inicializacija geometrije tarče.
        self.size = (int(round(region[2])), int(round(region[3])))
        self.position = (float(region[0] + region[2] / 2.0), float(region[1] + region[3] / 2.0))
        self.window = (
            int(round(self.size[0] * self.parameters.enlarge_factor)),
            int(round(self.size[1] * self.parameters.enlarge_factor)),
        )

        # Inicialni videz tarče: utežen histogram v izbranem barvnem prostoru.
        img_cs = self._to_color_space(image)
        patch, mask = get_patch(img_cs, self.position, self.size)
        self.hist_kernel = _epanechnikov_kernel_exact(
            patch.shape[1], patch.shape[0], sigma=self.parameters.hist_kernel_sigma
        )
        self.q = self._normalized_hist(patch, weights=self.hist_kernel * mask)

        # Po želji zgradimo še ozadni model (histogram obroča okoli tarče).
        self.b = None
        if self.parameters.use_background_model:
            bg_size = (
                int(round(self.size[0] * self.parameters.background_scale)),
                int(round(self.size[1] * self.parameters.background_scale)),
            )
            bg_patch, bg_mask = get_patch(img_cs, self.position, bg_size)
            bg_kernel = _epanechnikov_kernel_exact(bg_patch.shape[1], bg_patch.shape[0], sigma=1.0)
            # Obroč = veljavni del ozadnega obliža brez centralnega dela.
            ring = bg_mask * (1.0 - bg_kernel)
            self.b = self._normalized_hist(bg_patch, weights=ring)

    def track(self, image):
        # Začetek s pozicijo iz prejšnjega frame-a.
        img_cs = self._to_color_space(image)
        cx, cy = self.position
        tw, th = self.size
        max_shift = max(tw, th) / 2.0

        # Mean-shift iteracije lokalizacije.
        for _ in range(self.parameters.max_iters):
            # Lokalni obliž okoli trenutne ocene centra.
            patch, mask = get_patch(img_cs, (cx, cy), self.size)
            p = self._normalized_hist(patch, weights=self.hist_kernel * mask)

            # Razmerje cilj/aktualni histogram -> lookup uteži po barvnih binih.
            # Opomba: tukaj uporabimo v = q/(p+eps) (brez korena).
            # Na predavanjih se pogosto uporablja tudi sqrt(q/(p+eps)),
            # vendar v tem projektu ostanemo pri enostavnejši obliki.
            if self.b is None:
                v = self.q / (p + 1e-8)
            else:
                # Z ozadjem: kaznujemo barve, ki so pogoste v ozadju.
                v = self.q / (p + self.parameters.background_weight * self.b + 1e-8)
            backproj = backproject_histogram(patch, v, self.parameters.nbins).astype(np.float32)

            # Koordinate pikslov v lokalnem oknu (center = 0,0).
            h, w = backproj.shape[:2]
            ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
            xs -= (w - 1) / 2.0
            ys -= (h - 1) / 2.0

            # Mean-shift jedro.
            halfx = max((w - 1) / 2.0, 1.0)
            halfy = max((h - 1) / 2.0, 1.0)
            dist2 = (xs / halfx) ** 2 + (ys / halfy) ** 2
            ms_kernel = _kernel_weights(dist2, self.parameters.ms_kernel_type)

            # Končne uteži = backprojection * jedro * veljavna maska.
            weights = backproj * ms_kernel * mask
            s = float(np.sum(weights))
            if s <= 1e-12:
                # Brez uporabne mase ni smiselnega premika.
                break

            # Težiščni premik in omejitev premika na največ pol velikosti objekta.
            dx = float(np.sum(weights * xs) / s)
            dy = float(np.sum(weights * ys) / s)
            dx = float(np.clip(dx, -max_shift, max_shift))
            dy = float(np.clip(dy, -max_shift, max_shift))

            # Posodobi center in preveri konvergenco.
            cx_new = cx + dx
            cy_new = cy + dy
            shift = (dx * dx + dy * dy) ** 0.5
            cx, cy = cx_new, cy_new
            if shift < self.parameters.stop_eps:
                break

        # Shranimo novo stanje sledilnika.
        self.position = (cx, cy)

        # Posodobitev modela (lahko se izklopi z model_alpha = 0).
        patch_final, mask_final = get_patch(img_cs, self.position, self.size)
        q_new = self._normalized_hist(patch_final, weights=self.hist_kernel * mask_final)
        a = float(self.parameters.model_alpha)
        self.q = (1.0 - a) * self.q + a * q_new
        # Ponovna normalizacija po posodobitvi.
        s = float(np.sum(self.q))
        if s > 0:
            self.q /= s

        # Izhod v obliki [x, y, w, h] (zgornji levi kot + dimenzije).
        return [cx - tw / 2.0, cy - th / 2.0, float(tw), float(th)]
