from imutils.object_detection import non_max_suppression
import numpy as np
import time
import pytesseract as pt
import re
from tqdm import tqdm
import db_connect as db
import cv2
import os
import torch
from CRAFTS import craft
from CRAFTS import text_detector
from CRAFTS import refinenet
from collections import OrderedDict
from PIL import ImageEnhance, Image


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def load_craft():
    nets = craft.CRAFT()
    print("[INFO] loading CRAFT text detector...")
    nets.load_state_dict(copyStateDict(torch.load('CRAFTS/weights/craft_mlt_25k.pth', map_location='cpu')))

    print("[INFO] loading CRAFT REFINER...")
    refine_nets = refinenet.RefineNet()
    refine_nets.load_state_dict(
        copyStateDict(torch.load('CRAFTS/weights/craft_refiner_CTW1500.pth', map_location='cpu')))
    return nets, refine_nets


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
            img = cv2.resize(img, ((w // 32) * 32, (h // 32) * 32))
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
        result = cv2.resize(result, ((W // 32) * 32, (H // 32) * 32))

    return result


def string_process(text):
    # 9글자로 이루어진 의약품 코드 추출
    codes = re.findall('[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+', text)
    if codes:
        return codes
    codes = re.findall('[0-9]+', text)
    if codes:
        return codes
    return None


def extract_sub_info(res, sx, sy, ex, ey, img):
    crafts, refine = load_craft()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    image = img[sy:ey, sx:ex]
    (H, W) = image.shape[:2]
    if (H % 32) != 0 or (W % 32) != 0:
        if H < 32: H = 32
        if W < 32: W = 32
        image = cv2.resize(image, ((W // 32) * 32, (H // 32) * 32))
    image = cv2.pyrUp(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.resize(image, None, fx=1.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    boxes = text_detector.text_detect(crafts, refine, gray)

    #gray = line_proc(gray)

    print(boxes)

    cv2.imwrite(os.path.join('image', 'img' + time.strftime("%Y%m%d-%H%M%S") + '.jpg'), image)

    nums = [1, 1, 1]
    cnt = 0
    res = pt.image_to_string(gray, config='--psm 6 --oem 3', lang='kor')
    print(res)


def resize_apply_histo(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.resize(image, None, fx=1.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return gray

def extract_sub_info(res, sx, sy, ex, ey, img, crafts, refine):
    image = img[sy:ey, sx:ex]
    (H, W) = image.shape[:2]
    if (H % 32) != 0 or (W % 32) != 0:
        if H < 32: H = 32
        if W < 32: W = 32
        image = cv2.resize(image, ((W // 32) * 32, (H // 32) * 32))
    image = cv2.pyrUp(image)
    gray = resize_apply_histo(image)
    boxes = text_detector.text_detect(crafts, refine, gray)

    mp = {}
    for box in boxes:
        mp[box[0][0]] = box

    mp = sorted(mp.items())

    if len(mp) == 4:
        attrs = ['single_dose', 'daily_dose', 'total_dose']
        cnt = 0
        for i in range(1, len(mp)):
            (sx, sy), (ex, ey) = (int(mp[i][1][0][0]), int(mp[i][1][0][1])), (int(mp[i][1][2][0]), int(mp[i][1][2][1]))
            sub = gray[sy:ey, sx:ex]
            sub = cv2.resize(sub, None, fx=4.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            sub = cv2.cvtColor(sub, cv2.COLOR_BGR2RGB)
            sub = Image.fromarray(sub)
            sub = sub.convert("RGB")
            sub = ImageEnhance.Contrast(sub).enhance(3)
            sub = np.array(sub)
            sub = cv2.cvtColor(sub, cv2.COLOR_RGB2BGR)

            texts = pt.image_to_string(sub, config='--psm 6 --oem 3')
            try:
                texts = int(texts)
            except Exception as e:
                try :
                    texts = float(texts)
                except Exception as e2:
                    texts = 1

            res[0][attrs[cnt]] = texts
            print(attrs[cnt] , ' : ', texts)
            cnt+=1
            #print_img(sub)
    return res


def text_roi_extension(image, _startX, _endX, _startY, _endY, _W, _H, crafts, refine):
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
    text = pt.image_to_string(gray, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    res = string_process(text)
    if res:
        result = db.selectQuery(res)
        if result:
            (sx, sy), (ex, ey) = (_endX, _startY), (_W, _endY)
            result = extract_sub_info(result, sx, sy, ex, ey, image, crafts, refine)
            return result

    # x축 증가
    for i in range(5, 40, 5):
        if _endX + i >= _W:
            break
        img = cv2.pyrUp(image[_startY - (i // 50):_endY + (i // 50), _startX:_endX + i])

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        text = pt.image_to_string(gray, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
        # print_img(gray)
        # print(text)
        res = string_process(text)
        if res:
            result = db.selectQuery(res)
            if result:
                (sx, sy), (ex, ey) = (_endX + i, _startY - (i // 10)), (_W, _endY + (i // 10))
                result = extract_sub_info(result, sx, sy, ex, ey, image, crafts, refine)
                return result
        else:
            break

    # x축 감소
    for i in range(5, 40, 5):
        if _startX - i < 0:
            break
        img = cv2.pyrUp(image[_startY - (i // 10):_endY + (i // 10), _startX - i: _endX])

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        text = pt.image_to_string(gray, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
        # print_img(gray)
        # print(text)
        res = string_process(text)
        if res:
            result = db.selectQuery(res)
            if result:
                (sx, sy), (ex, ey) = (_endX, _startY - (i // 10)), (_W, _endY + (i // 10))
                result = extract_sub_info(result, sx, sy, ex, ey, image, crafts, refine)
                return result
        else:
            break

    # 4방향 동시 증가
    for i in range(5, 20, 5):
        img = cv2.pyrUp(
            image[max(int(_startY - i // 3), 0):min(int(_endY + i // 3), _H), max(_startX - i, 0):min(_endX + i, _W)])

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        text = pt.image_to_string(gray, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
        # print_img(gray)
        # print(text)
        res = string_process(text)
        if res:
            result = db.selectQuery(res)
            if result:
                (sx, sy), (ex, ey) = (_endX + i, max(int(_startY - i // 3), 0)), (_W, min(int(_endY + i // 3), _H))
                result = extract_sub_info(result, sx, sy, ex, ey, image, crafts, refine)
                return result
        else:
            break
    return None


def find_roi(image, min_thresh, max_thresh, net):
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

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

    for y in range(0, numRows):
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

            if (endX - startX) < min_thresh or (endX - startX) > max_thresh:
                continue
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return rects, confidences


def text_detect(image, net, crafts, refine):
    (H, W) = image.shape[:2]
    rects, confidences = find_roi(image, 40, 400, net)
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    ret = {'drugs': []}
    origin = image.copy()
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(origin, (startX, startY), (endX, endY), (0, 255, 0), 2)

    for (startX, startY, endX, endY) in tqdm(boxes):
        try:
            res = text_roi_extension(image, startX, endX, startY, endY, W, H, crafts, refine)
            if res and res[0]['code'] not in ret['drugs']:
                ret['drugs'].append(res[0])
        # draw the bounding box on the image
        except Exception as ex:
            print('error report : ', ex)
    cv2.imwrite(os.path.join('image', 'result.jpg'), origin)
    return ret


def hough_linep(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
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

            if abs(thetas) < 1:
                return img
            else:
                return rotate_theta_deg(img, thetas)
    return img


def line_elimination(image):
    # Line Elimination
    temp = image
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
    return temp


def line_proc(img):
    C = 5
    blk_size = 9
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk_size, C)
    # LPF
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = line_elimination(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    print_img(img)
    return img


def image_warp(src_loc, net, crafts, refine):
    # 이미지 읽기
    img = cv2.imread(src_loc)
    print(img.shape)
    img, rot = image_resize(img)
    img = hough_linep(img)
    print(img.shape)
    img = edge_detect(img)
    print(img.shape)

    if rot:
        res = text_detect(img, net, crafts, refine)
        if res:
            return res
        img = rotate_180deg(img)
        return text_detect(img, net, crafts, refine)

    else:
        return text_detect(img, net, crafts, refine)


if __name__ == '__main__':
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet('frozen_east_text_detection.pb')
    crafts, refine = load_craft()
    print(image_warp('image/test7.jpg', net, crafts, refine))

