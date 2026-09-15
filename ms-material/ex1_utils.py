import cv2


def gausssmooth(image, sigma):
    sigma = float(max(0.0, sigma))
    if sigma <= 0:
        return image
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
