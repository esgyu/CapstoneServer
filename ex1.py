import cv2
import numpy as np
import pytesseract as pt

def image_warp(src_loc):
    win_name = 'scan'
    # 이미지 읽기
    img = cv2.imread(src_loc)
    img = cv2.resize(img, None, fx=0.25, fy=0.2, interpolation=cv2.INTER_AREA)
    draw = img.copy()

    # 그레이스 스케일 변환 및 케니 엣지
    result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    result = clahe.apply(result)
    # printimg('CLAHE', result)

    # Binary Image Elimination
    ret, result = cv2.threshold(result, 100, 255, cv2.THRESH_BINARY)
    #printimg('Binary Image', result)

    # LPF
    result = cv2.bilateralFilter(result,9, 75, 75)
    #printimg('LPF', result)

    # Sobel Mask로 Line Elimiation 필요...
    # sobel_x = cv2.Sobel(result, cv2.CV_64F, 1, 0, ksize=3)
    # sobel_x = cv2.convertScaleAbs(sobel_x)
    # sobel_y = cv2.Sobel(result, cv2.CV_64F, 0, 1, ksize=3)
    # sobel_y = cv2.convertScaleAbs(sobel_y)
    # cv2.imshow('sobel', sobel_x)
    # cv2.waitKey(0)
    # cv2.imshow('sobel', sobel_y)
    # cv2.waitKey(0)

    text = pt.image_to_string(result, config='--psm 6', lang='kor')
    print("================ OCR result ================")
    print(text)

def printimg(label, img):
    cv2.imshow(label, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image_warp('KakaoTalk_20200509_001508967.jpg')