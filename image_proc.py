import cv2
import numpy as np
import pytesseract as pt
import re
import db_connect as db

def image_warp(src_loc):
    # 이미지 읽기
    img = cv2.imread(src_loc)
    height, width = img.shape[:2]
    if width*4 != height*3 and width*1.2 > height:
        (h, w) = img.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), 270, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        img = cv2.warpAffine(img, M, (nW, nH))

    if height>2560 and width > 1920:
        img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

    draw = img.copy()
    # 그레이스 스케일 변환 및 케니 엣지
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0) # 가우시안 블러로 노이즈 제거
    iswarp = False
    try:
        edged = cv2.Canny(gray, 75, 200)    # 케니 엣지로 경계 검출
        #printimg('Gray Scale & Canny Edge', edged)
        # 컨투어 찾기
        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, \
                                                        cv2.CHAIN_APPROX_SIMPLE)
        # 모든 컨투어 그리기
        cv2.drawContours(draw, cnts, -1, (0,255,0))
        #printimg('Draw Contour', draw)
        # 컨투어들 중에 영역 크기 순으로 정렬

        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
        for c in cnts:
            # 영역이 가장 큰 컨투어 부터 근사 컨투어 단순화
            peri = cv2.arcLength(c, True)   # 둘레 길이
            # 둘레 길이의 0.02 근사값으로 근사화
            vertices = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(vertices) == 4: # 근사한 꼭지점이 4개면 중지
                break
        pts = vertices.reshape(4, 2) # N x 1 x 2 배열을 4 x 2크기로 조정
        for x,y in pts:
            cv2.circle(draw, (x,y), 10, (0,255,0), -1) # 좌표에 초록색 동그라미 표시

        #printimg('4 Points', draw)
        # 좌표 4개 중 상하좌우 찾기 ---②
        sm = pts.sum(axis=1)                 # 4쌍의 좌표 각각 x+y 계산
        diff = np.diff(pts, axis = 1)       # 4쌍의 좌표 각각 x-y 계산

        topLeft = pts[np.argmin(sm)]         # x+y가 가장 값이 좌상단 좌표
        bottomRight = pts[np.argmax(sm)]     # x+y가 가장 큰 값이 좌상단 좌표
        topRight = pts[np.argmin(diff)]     # x-y가 가장 작은 것이 우상단 좌표
        bottomLeft = pts[np.argmax(diff)]   # x-y가 가장 큰 값이 좌하단 좌표

        # 변환 전 4개 좌표
        pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

        # 변환 후 영상에 사용할 서류의 폭과 높이 계산 ---③
        w1 = abs(bottomRight[0] - bottomLeft[0])    # 상단 좌우 좌표간의 거리
        w2 = abs(topRight[0] - topLeft[0])          # 하당 좌우 좌표간의 거리
        h1 = abs(topRight[1] - bottomRight[1])      # 우측 상하 좌표간의 거리
        h2 = abs(topLeft[1] - bottomLeft[1])        # 좌측 상하 좌표간의 거리
        width = max([w1, w2])                       # 두 좌우 거리간의 최대값이 서류의 폭
        height = max([h1, h2])                      # 두 상하 거리간의 최대값이 서류의 높이

        # 검출 좌표 길이가 200 * 200 미만이면 원본이미지를 Gray Scaling함
        if width > 200 and height > 200 :
            # 변환 후 4개 좌표
            pts2 = np.float32([[0,0], [width-1,0], [width-1,height-1], [0,height-1]])
            # 변환 행렬 계산
            mtrx = cv2.getPerspectiveTransform(pts1, pts2)
            # 원근 변환 적용
            result = cv2.warpPerspective(img, mtrx, (width, height))
            iswarp = True
            #printimg('Warping', result)
        else:
            result = gray
    except Exception:
        result = gray

    # Warping 이미지 GrayScaling 후 CLAHE로 Histogram Eqaulization
    if iswarp:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    result = clahe.apply(result)
    #printimg('CLAHE', result)
    # Binary Image Elimination
    C = 5
    blk_size = 9
    result = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blk_size, C)
    #printimg('Adaptive Binary Image', result)
    # LPF
    result = cv2.bilateralFilter(result,9, 75, 75)
    #printimg('LPF', result)


    # Line Elimination
    temp = result
    temp = cv2.bitwise_not(temp)
    th2 = cv2.adaptiveThreshold(temp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    horizontal = th2
    vertical = th2
    rows, cols = horizontal.shape

    horizontalsize = int(cols / 30)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
    horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))

    verticalsize = int(rows / 30)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))

    sum = cv2.add(vertical, horizontal)
    temp = cv2.absdiff(sum, temp)

    temp = 255 - temp
    temp = cv2.bilateralFilter(temp, 9, 75, 75)
    result = temp

    text = pt.image_to_string(result, config='--psm 6', lang='kor')
    print("================ OCR result ================")
    return stringProcess(text)

def stringProcess(text):
    # 9글자로 이루어진 의약품 코드 추출
    codes = re.findall('[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+',text)
    ret = {'drugs':[]}
    for code in codes:
        result = db.selectQuery(code)
        print(code)
        if result:
            ret['drugs'].append(result[0])
            print(result)
        else :
            print("Fail!")
    print(ret)
    return ret

def printimg(label, img):
    cv2.imshow(label, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image_warp('KakaoTalk_20200510_161622423.jpg')