from imutils.object_detection import non_max_suppression
import numpy as np
import time
import pytesseract as pt
import re
from tqdm import tqdm
import db_connect as db
import cv2


def print_img(img):
    cv2.imshow('aaa', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_resize(img):
    height, width = img.shape[:2]
    rot = False
    if width * 3 == height * 4:
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
        rot = True
    if height > 1920 and width > 1440:
        img = cv2.resize(img, (1440, 1920))
    else:
        (h, w) = img.shape[:2]
        if (height % 32) != 0 or (width % 32) != 0:
            img = cv2.resize(img, ((w//32)*32, (h//32)*32))
    return img, rot


def rotate_180deg(img):
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
    return img


def rotate_theta_deg(img, theta):
    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), theta, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    img = cv2.warpAffine(img, M, (nW, nH))
    return img


def edge_detect(img):
    (H, W) = img.shape[:2]
    # 그레이스 스케일 변환 및 케니 엣지
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # 가우시안 블러로 노이즈 제거
    try:
        edged = cv2.Canny(gray, 75, 200)  # 케니 엣지로 경계 검출
        # 컨투어 찾기
        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 컨투어들 중에 영역 크기 순으로 정렬
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        for c in cnts:
            # 영역이 가장 큰 컨투어 부터 근사 컨투어 단순화
            peri = cv2.arcLength(c, True)  # 둘레 길이
            # 둘레 길이의 0.02 근사값으로 근사화
            vertices = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(vertices) == 4:  # 근사한 꼭지점이 4개면 중지
                break
        pts = vertices.reshape(4, 2)  # N x 1 x 2 배열을 4 x 2크기로 조정

        # 좌표 4개 중 상하좌우 찾기 ---②
        sm = pts.sum(axis=1)  # 4쌍의 좌표 각각 x+y 계산
        diff = np.diff(pts, axis=1)  # 4쌍의 좌표 각각 x-y 계산

        topLeft = pts[np.argmin(sm)]  # x+y가 가장 값이 좌상단 좌표
        bottomRight = pts[np.argmax(sm)]  # x+y가 가장 큰 값이 좌상단 좌표
        topRight = pts[np.argmin(diff)]  # x-y가 가장 작은 것이 우상단 좌표
        bottomLeft = pts[np.argmax(diff)]  # x-y가 가장 큰 값이 좌하단 좌표

        # 변환 전 4개 좌표
        pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

        # 변환 후 영상에 사용할 서류의 폭과 높이 계산 ---③
        w1 = abs(bottomRight[0] - bottomLeft[0])  # 상단 좌우 좌표간의 거리
        w2 = abs(topRight[0] - topLeft[0])  # 하당 좌우 좌표간의 거리
        h1 = abs(topRight[1] - bottomRight[1])  # 우측 상하 좌표간의 거리
        h2 = abs(topLeft[1] - bottomLeft[1])  # 좌측 상하 좌표간의 거리
        width = max([w1, w2])  # 두 좌우 거리간의 최대값이 서류의 폭
        height = max([h1, h2])  # 두 상하 거리간의 최대값이 서류의 높이

        # 검출 좌표 길이가 원본이미지의 60%보다 작으면 잘못된 Warping으로 판단
        if width > W * 0.6 and height > H * 0.6:
            # 변환 후 4개 좌표
            pts2 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
            # 변환 행렬 계산
            mtrx = cv2.getPerspectiveTransform(pts1, pts2)
            # 원근 변환 적용
            result = cv2.warpPerspective(img, mtrx, (width, height))
        else:
            result = img
    except Exception:
        result = img

    H, W = result.shape[:2]

    # Warping 후 n*32 x m*32 사이즈가 아닌경우 resizing
    if (H % 32) != 0 or (W % 32) != 0:
        result = cv2.resize(result, ((W//32)*32, (H//32)*32))

    return result


def string_process(text):
    # 9글자로 이루어진 의약품 코드 추출
    codes = re.findall('[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+', text)
    if codes:
        return codes
    codes = re.findall('[0-9]+[0-9]+[0-9]+', text)
    if codes:
        return 'maybe'
    codes = re.findall('[0-9]+', text)
    if codes:
        return 'maybe'
    return None


def extract_sub_info(res, sx, sy, ex, ey, img, code):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    image = img[sy:ey, sx:ex]
    (H, W) = image.shape[:2]
    if (H%32)!=0 or (W%32)!=0:
        image = cv2.resize(image, ((W//32)*32, (H//32)*32))
    image = cv2.pyrUp(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res = pt.image_to_string(gray, config='--psm 6', lang='kor')
    texts = res.split('\n')
    for text in texts:
        print(text , '끝')
    print_img(gray)


def text_roi_extension(image, _startX, _endX, _startY, _endY, _W, _H):
    _startX = max(_startX - 5, 0)
    _endX = min(_endX + 5, _W)
    _startY = max(_startY - 5, 0)
    _endY = min(_endY + 10, _H)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    img = image[_startY:_endY, _startX:_endX]

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pt.image_to_string(gray, config='--psm 6', lang='kor')
    res = string_process(text)

    if not res:
        return None
    result = db.selectQuery(res)
    if result:
        (sx, sy), (ex, ey) = (_startX, _startY), (_W, _endY)
        extract_sub_info(result, sx, sy, ex, ey, image, result[0]['code'])
        return result

    # x축 증가
    for i in range(10, 60, 10):
        if _endX + i >= _W:
            break
        img = cv2.pyrUp(image[_startY - (i // 50):_endY + (i // 50), _startX:_endX + i])

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        text = pt.image_to_string(gray, config='--psm 6', lang='kor')
        #print_img(gray)
        print(text)
        res = string_process(text)
        if res:
            result = db.selectQuery(res)
            if result:
                (sx, sy), (ex, ey) = (_startX, _startY - (i // 50)), (_W, _endY + (i // 50))
                extract_sub_info(result, sx, sy, ex, ey, image, result[0]['code'])
                return result
        else:
            break

    # x축 감소
    for i in range(10, 60, 10):
        if _startX - i < 0:
            break
        img = cv2.pyrUp(image[_startY - (i // 50):_endY + (i // 50), _startX - i: _endX])

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        text = pt.image_to_string(gray, config='--psm 6', lang='kor')
        #print_img(gray)
        print(text)
        res = string_process(text)
        if res:
            result = db.selectQuery(res)
            if result:
                (sx, sy), (ex, ey) = (_startX-i, _startY - (i // 50)), (_W, _endY + (i // 50))
                extract_sub_info(result, sx, sy, ex, ey, image, result[0]['code'])
                return result
        else:
            break
    # 4방향 동시 증가
    for i in range(10, 60, 10):
        img = cv2.pyrUp(
            image[max(int(_startY - i // 2), 0):min(int(_endY + i // 2), _H), max(_startX - i, 0):min(_endX + i, _W)])

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        text = pt.image_to_string(gray, config='--psm 6', lang='kor')
        #print_img(gray)
        print(text)
        res = string_process(text)
        if res:
            result = db.selectQuery(res)
            if result:
                (sx, sy), (ex, ey) = (max(_startX-i, 0), max(int(_startY - i // 2), 0)), (_W, min(int(_endY + i // 2), _H))
                extract_sub_info(result, sx, sy, ex, ey, image, result[0]['code'])
                return result
        else:
            break

    # 4방향 동시 감소
    for i in range(1, 10):
        img = cv2.pyrUp(image[(_startY + i):(_endY - i), (_startX+i):(_endX - i)])

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        text = pt.image_to_string(gray, config='--psm 6', lang='kor')
        #print_img(gray)
        print(text)
        res = string_process(text)
        if res:
            result = db.selectQuery(res)
            if result:
                (sx, sy), (ex, ey) = (_startX+i, (_startY + i)), (_W, (_endY - i), img)
                extract_sub_info(result, sx, sy, ex, ey, image, result[0]['code'])
                return result
        else:
            break
    return None


def find_roi(image, min, max):
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet('frozen_east_text_detection.pb')

    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)

    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    print("[INFO] text detection took {:.6f} seconds".format(end - start))

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            # min_confidence default 0.5
            if scoresData[x] < 0.3:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            if (endX - startX) < min or (endX - startX) > max:
                continue
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return rects, confidences


def text_detect(image):
    (H, W) = image.shape[:2]
    rects, confidences = find_roi(image, 40, 400)
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    ret = {'drugs': []}
    origin = image.copy()
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(origin, (startX, startY), (endX, endY), (0, 255, 0), 2)

    for (startX, startY, endX, endY) in tqdm(boxes):
        try:
            res = text_roi_extension(image, startX, endX, startY, endY, W, H)
            if res and res[0]['code'] not in ret['drugs']:
                ret['drugs'].append(res[0])
        # draw the bounding box on the image
        except Exception as ex:
            print('error report : ', ex)
    cv2.imwrite('result.jpg', origin)
    return ret


def hough_linep(img):
    (H, W) = img.shape[:2]
    if (H, W) > (1280, 960):
        img = cv2.resize(img, (810, 1080))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    for thresh in range(600, 200, -50):
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 300)
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            distdiag = np.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))
            distx = abs(x1-x2)

            thetas = np.rad2deg(np.arccos(distx/distdiag))
            if thetas > 30:
                continue

            if y1 > y2:
                thetas *=-1

            if abs(thetas) < 1:
                return img
            else:
                return rotate_theta_deg(img, thetas)
    return img


def image_warp(src_loc):
    # 이미지 읽기
    img = cv2.imread(src_loc)
    print(img.shape)
    img, rot = image_resize(img)
    img = hough_linep(img)
    img = edge_detect(img)
    #print_img(img)
    if rot:
        res = text_detect(img)
        if res:
            return res
        img = rotate_180deg(img)
        return text_detect(img)

    else:
        return text_detect(img)


if __name__ == '__main__':
    print(image_warp('20200602-173535_Android_Flask_.jpg'))
