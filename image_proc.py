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
from crafts import craft
from crafts import text_detector
from crafts import refinenet
from collections import OrderedDict
from PIL import ImageEnhance, Image
import glob

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


# 이미지 Resizing 및 1920*1440 이상 이미지를 1920*1440으로, 그 외의 경우는 높이/너비를 32배수로 맞춤
# 높이-너비 비율이 4:3이 아닌경우 시계방향 90도 회전
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


# 이미지가 시계방향으로 90도 돌아가서 전송되는 경우 재회전
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


# HoughLine으로 검출된 수평선의 각도에 따라 수평을 맞추기 위해 원하는 만큼 회전
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


# 주어진 사진이 명확하게 처방전의 가장 바깥쪽 직선을 포함한 경우 Warping
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


# 정규표현식을 통해 9글자 코드를 추출, 숫자가 한글자도 없는 경우 약물 코드로 판단하지 않음
def string_process(text):
    codes = re.findall('[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+', text)
    if codes:
        return codes
    codes = re.findall('[0-9]+', text)
    if codes:
        return codes
    return None


# CLAHE Histogram Equalization 진행 후 높이 2배 향상하여 반환
def resize_apply_histo(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.resize(image, None, fx=1.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return gray


def extract_sub_info(sx, sy, ex, ey, img, crafts, refine):
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

    if len(mp) > 1:
        mp = sorted(mp.items())
        return mp[1:]
    return None


def text_roi_extension_proc(image, _startY, _endY, _startX, _endX):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img = cv2.pyrUp(image[_startY:_endY, _startX:_endX])
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def text_roi_extension(image, _startX, _endX, _startY, _endY, _W, _H, _crafts, _refine):
    _startX = max(_startX - 5, 0)
    _endX = min(_endX + 5, _W)
    _startY = max(_startY - 5, 0)
    _endY = min(_endY + 10, _H)
    gray = text_roi_extension_proc(image, _startY, _endY, _startX, _endX)
    text = pt.image_to_string(gray, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    res = string_process(text)
    if res:
        result = db.selectQuery(res)
        if result:
            (sx, sy), (ex, ey) = (_endX, _startY), (_W, _endY)
            #result = extract_sub_info(result, sx, sy, ex, ey, image, crafts, refine)
            mp = extract_sub_info(sx, sy, ex, ey, image, _crafts, _refine)
            return result, mp, ((sx, sy), (ex, ey))

    # x축 증가
    for i in range(5, 40, 5):
        if _endX + i >= _W:
            break
        gray = text_roi_extension_proc(image, _startY - (i // 50), _endY + (i // 50), _startX, _endX + i)
        text = pt.image_to_string(gray, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
        res = string_process(text)
        if res:
            result = db.selectQuery(res)
            if result:
                (sx, sy), (ex, ey) = (_endX + i, _startY - (i // 10)), (_W, _endY + (i // 10))
                mp = extract_sub_info(sx, sy, ex, ey, image, _crafts, _refine)
                return result, mp, ((sx, sy), (ex, ey))
        else:
            break

    # x축 감소
    for i in range(5, 40, 5):
        if _startX - i < 0:
            break
        gray = text_roi_extension_proc(image, _startY - (i // 10), _endY + (i // 10), _startX - i, _endX)
        text = pt.image_to_string(gray, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
        res = string_process(text)
        if res:
            result = db.selectQuery(res)
            if result:
                (sx, sy), (ex, ey) = (_endX, _startY - (i // 10)), (_W, _endY + (i // 10))
                mp = extract_sub_info(sx, sy, ex, ey, image, _crafts, _refine)
                return result, mp, ((sx, sy), (ex, ey))
        else:
            break

    # 4방향 동시 증가
    for i in range(5, 20, 5):
        gray = text_roi_extension_proc(image, max(int(_startY - i // 3), 0), min(int(_endY + i // 3), _H),
                                        max(_startX - i, 0), min(_endX + i, _W))
        text = pt.image_to_string(gray, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
        res = string_process(text)
        if res:
            result = db.selectQuery(res)
            if result:
                (sx, sy), (ex, ey) = (_endX + i, max(int(_startY - i // 3), 0)), (_W, min(int(_endY + i // 3), _H))
                mp = extract_sub_info(sx, sy, ex, ey, image, _crafts, _refine)
                return result, mp, ((sx, sy), (ex, ey))
        else:
            break
    return None, None, None


def find_roi(image, min_thresh, max_thresh, net):
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    (H, W) = image.shape[:2]
    print(H, W)
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)

    #start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    #end = time.time()

    #print("[INFO] text detection took {:.6f} seconds".format(end - start))

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


def crop_image(img, sx, sy, ex, ey):
    image = img[sy:ey, sx:ex]
    (H, W) = image.shape[:2]
    if (H % 32) != 0 or (W % 32) != 0:
        if H < 32: H = 32
        if W < 32: W = 32
        image = cv2.resize(image, ((W // 32) * 32, (H // 32) * 32))
    image = cv2.pyrUp(image)
    gray = resize_apply_histo(image)
    return gray


def proc_dose(image, ret, mp, point):
    single_dose_point = None
    daily_dose_point = None
    total_dose_point = None

    for obj in ret['drugs']:
        code = obj['code']
        if len(mp[code]) == 3:
            single_dose_point = mp[code][0][1][0][0]
            daily_dose_point = mp[code][1][1][0][0]
            total_dose_point = mp[code][2][1][0][0]
        break

    if not single_dose_point:
        (H, W) = image.shape[:2]
        for obj in ret['drugs']:
            code = obj['code']
            if len(mp[code]) == 1:
                x1 = mp[code][0][1][0][0]
                if not single_dose_point:
                    single_dose_point = x1
                else:
                    if abs(single_dose_point) < 20:
                        continue
                    if single_dose_point < x1:
                        if not daily_dose_point:
                            daily_dose_point = x1
                        else:
                            if abs(daily_dose_point - x1) < 20:
                                continue
                            else:
                                if daily_dose_point < x1:
                                    if not total_dose_point:
                                        total_dose_point = x1
                                else:
                                    tmp = daily_dose_point
                                    daily_dose_point = x1
                                    x1 = tmp
                                    if not total_dose_point:
                                        total_dose_point = x1
                    else:
                        tmp = single_dose_point
                        single_dose_point = x1
                        x1 = tmp
                        if not daily_dose_point:
                            daily_dose_point = x1
                        else:
                            if abs(daily_dose_point - x1) < 20:
                                continue
                            else:
                                if daily_dose_point < x1:
                                    if not total_dose_point:
                                        total_dose_point = x1
                                else:
                                    tmp = daily_dose_point
                                    daily_dose_point = x1
                                    x1 = tmp
                                    if not total_dose_point:
                                        total_dose_point = x1
            elif len(mp[code]) == 2:
                x1 = mp[code][0][1][0][0]
                x2 = mp[code][1][1][0][0]
                if not single_dose_point:
                    single_dose_point = x1
                    if not daily_dose_point:
                        daily_dose_point = x2
                    else:
                        if (daily_dose_point - x2) < 20:
                            continue
                        else:
                            if daily_dose_point < x2:
                                if not total_dose_point:
                                    total_dose_point = x2
                            else:
                                tmp = daily_dose_point
                                daily_dose_point = x2
                                x2 = tmp
                                if not total_dose_point:
                                    total_dose_point = x2
                else:
                    if abs(single_dose_point-x1) < 20:
                        if not daily_dose_point:
                            daily_dose_point = x2
                        else:
                            if abs(daily_dose_point-x2) < 20:
                                continue
                            if daily_dose_point < x2:
                                if not total_dose_point:
                                    total_dose_point = x2
                            else:
                                tmp = daily_dose_point
                                daily_dose_point = x2
                                x2 = tmp
                                if not total_dose_point:
                                    total_dose_point = x2
                    else:
                        if single_dose_point < x1:
                            if not daily_dose_point:
                                daily_dose_point = x2
                                if not total_dose_point:
                                    total_dose_point = x2
                            else:
                                if abs(x1-daily_dose_point) < 20:
                                    if not total_dose_point:
                                        total_dose_point = x2
                        else:
                            tmp = single_dose_point
                            single_dose_point = x1
                            x1 = tmp
                            if not daily_dose_point:
                                daily_dose_point = x1
                                if not total_dose_point:
                                    total_dose_point = x2
                            else:
                                if not total_dose_point:
                                    total_dose_point = x2

    if not single_dose_point or not daily_dose_point or not total_dose_point:
        return ret

    attrs = ['single_dose', 'daily_dose', 'total_dose']
    for obj in ret['drugs']:
        code = obj['code']
        (_sx, _sy), (_ex, _ey) = point[code]
        gray = crop_image(image, _sx, _sy, _ex, _ey)
        for i in range(len(mp[code])):
            (sx, sy), (ex, ey) = (int(mp[code][i][1][0][0]), int(mp[code][i][1][0][1])), (int(mp[code][i][1][2][0]), int(mp[code][i][1][2][1]))
            diff = abs(sx - single_dose_point)
            cnt = 0
            if diff > abs(sx-daily_dose_point):
                diff = abs(sx-daily_dose_point)
                cnt=1
            if diff > abs(sx-total_dose_point):
                cnt=2

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
                try:
                    texts = float(texts)
                except Exception as e2:
                    texts = 1
            target = next((item for item in ret['drugs'] if item['code'] == code), None)
            target[attrs[cnt]] = texts
            #print(code, ' - ' , attrs[cnt], ' : ', texts)
    return ret


def text_detect(image, net, crafts, refine):
    (H, W) = image.shape[:2]
    rects, confidences = find_roi(image, 40, 400, net)
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    ret = {'drugs': []}
    mp = {}
    point = {}
    origin = image.copy()
    for (startX, startY, endX, endY) in boxes:
        cv2.rectangle(origin, (startX, startY), (endX, endY), (0,255,0), 2)
    for (startX, startY, endX, endY) in boxes:
        try:
            res, tmp, pt = text_roi_extension(image, startX, endX, startY, endY, W, H, crafts, refine)
            if res and res[0]['code'] not in ret['drugs']:
                ret['drugs'].append(res[0])
                if tmp:
                    mp[res[0]['code']] = tmp
                point[res[0]['code']] = pt
        # draw the bounding box on the image
        except Exception as ex:
            print('error report : ', ex)
    cv2.imwrite('result.jpg', origin)
    ret = proc_dose(image, ret, mp, point)
    if not ret:
        del image
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

height = [320, 640, 1024, 2048, 4096, 8192, 4096, 4544]
width = [320, 640, 1024, 2048, 4096, 4096, 30240, 30272]

def image_warp(src_loc, net, crafts, refine):
    # 이미지 읽기
    img = cv2.imread(src_loc)
    img, rot = image_resize(img)
    img = hough_linep(img)
    img = edge_detect(img)
    if rot:
        res = text_detect(img, net, crafts, refine)
        if res:
            return res
        img = rotate_180deg(img)
        return text_detect(img, net, crafts, refine)

    else:
        return text_detect(img, net, crafts, refine)


def process(net, crafts, refine):
    files = glob.glob('image/test/*.jpg')
    cnt = 0
    codecnt = 0
    singlecnt = 0
    dailycnt = 0
    totalcnt = 0
    total_document = 0
    for file in tqdm(files):
        lists = re.findall(r'\[[0-9].*[0-9]\]', file)
        objs = lists[0].split('],')
        result = {'drugs': []}
        for obj in objs:
            obj = obj.replace('[', '')
            obj = obj.replace(']', '')
            obj = obj.replace(' ', '')
            info = obj.split(',')
            medi = {'code': info[0], 'single_dose': info[1], 'daily_dose': info[2], 'total_dose': info[3]}
            result['drugs'].append(medi)
        try:
            ret = image_warp(file, net, crafts, refine)
        except Exception as e:
            print('error report :', e)
            continue
        total_document += 1
        cnt += len(result['drugs'])
        for obj in ret['drugs']:
            code = str(obj['code'])
            single = str(int(float(obj['single_dose'])))
            daily = str(obj['daily_dose'])
            total = str(obj['total_dose'])
            target = next((item for item in result['drugs'] if item['code'] == code), None)
            if not target: continue
            codecnt += 1
            if target['single_dose'] == single:
                singlecnt += 1
            if target['daily_dose'] == daily:
                dailycnt += 1
            if target['total_dose'] == total:
                totalcnt += 1

    print(cnt, codecnt, singlecnt, dailycnt, totalcnt)
    print('Total Processed Document : ', total_document)
    print('Code Accuracy :', (codecnt / cnt) * 100)
    print('Single Dose Accuracy :', (singlecnt / cnt) * 100)
    print('Daily Dose Accuracy :', (dailycnt / cnt) * 100)
    print('Total Dose Accuracy :', (totalcnt / cnt) * 100)

if __name__ == '__main__':
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet('frozen_east_text_detection.pb')
    crafts, refine = load_craft()
    process(net, crafts, refine)
