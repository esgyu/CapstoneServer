import glob
import cv2
import numpy as np
from scipy.ndimage import interpolation as inter

def hough_linep(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    for thresh in range(600, 200, -50):
        lines = cv2.HoughLines(edges, 1, np.pi / 180, thresh)
        if lines is None: continue
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)

            distdiag = np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
            distx = abs(x1 - x2)

            thetas = np.rad2deg(np.arccos(distx / distdiag))
            if thetas > 30:
                continue

            if y1 > y2:
                thetas *= -1


            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, thetas, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
    return img


if __name__ == '__main__':
    for file in glob.glob('*.jpg'):
        image = cv2.imread(file)
        rotated = hough_linep(image)
        cv2.imwrite('numbers/rotated_' + file, rotated)

