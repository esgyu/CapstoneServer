import numpy as np
import pytesseract as pt
import re
from tqdm import tqdm
import db_connect as db
import cv2
import torch
from crafts import craft
from crafts import text_detector
from crafts import refinenet
from collections import OrderedDict
from PIL import ImageEnhance, Image
import glob
from typing import Tuple, Union
import math
from deskew import determine_skew

def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)


def hough_linep(img):
    img = histo_equlization(img)

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
            if thetas > 30 or thetas < 0.5:
                continue

            if y1 > y2:
                thetas *= -1

            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, thetas, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            return img
    return img


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
    if width * 3 == height * 4 or height * 16 == width * 9:
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


# 정규표현식을 통해 9글자 코드를 추출, 숫자가 한글자도 없는 경우 약물 코드로 판단하지 않음
def string_process(text):
    codes = re.findall('[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+', text)
    if codes:
        return codes[0]
    return None


def histo_equlization(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    gray = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return gray


def insert_dose(dose, text, _x1, _x3, x3):
    diff = _x3 - _x1
    if '1회' in text or '투약량' in text:
        if '투여' in text or '횟수' in text or '1일' in text:
            ndiff = (diff//5)
            diff = ndiff*3
            if not dose['single_dose']:
                dose['single_dose'].append([_x1 + x3, _x1 + diff + x3])
            else:
                dose['single_dose'][0][0] = min(dose['single_dose'][0][0], _x1 + x3)
                dose['single_dose'][0][1] = max(dose['single_dose'][0][1], _x1 + diff + x3)
            _x1 += diff
            diff = ndiff*2
            if not dose['daily_dose']:
                dose['daily_dose'].append([_x1 + x3, _x1 + diff + x3])
            else:
                dose['daily_dose'][0][0] = min(dose['daily_dose'][0][0], _x1 + x3)
                dose['daily_dose'][0][1] = max(dose['daily_dose'][0][1], _x1 + diff + x3)
        else:
            if not dose['single_dose']:
                dose['single_dose'].append([_x1 + x3, _x1 + diff + x3])
            else:
                dose['single_dose'][0][0] = min(dose['single_dose'][0][0], _x1 + x3)
                dose['single_dose'][0][1] = max(dose['single_dose'][0][1], _x1 + diff + x3)
        return True
    if '1일' in text or '투여' in text or '횟수' in text:
        if '일수' in text or '투약' in text:
            diff = diff//2
            if not dose['daily_dose']:
                dose['daily_dose'].append([_x1 + x3+3, _x1 + diff + x3])
            else:
                dose['daily_dose'][0][0] = min(dose['daily_dose'][0][0], _x1 + x3+3)
                dose['daily_dose'][0][1] = max(dose['daily_dose'][0][1], _x1 + diff + x3)
            _x1 += diff
            if not dose['total_dose']:
                dose['total_dose'].append([_x1 + x3, _x1 + diff + x3 + 10])
            else:
                dose['total_dose'][0][0] = min(dose['total_dose'][0][0], _x1 + x3)
                dose['total_dose'][0][1] = max(dose['total_dose'][0][1], _x1 + diff + x3 + 10)
        else:
            if not dose['daily_dose']:
                dose['daily_dose'].append([_x1 + x3, _x1 + diff + x3])
            else:
                dose['daily_dose'][0][0] = min(dose['daily_dose'][0][0], _x1 + x3)
                dose['daily_dose'][0][1] = max(dose['daily_dose'][0][1], _x1 + diff + x3)
        return True
    if '총' in text or '일수' in text or ('투약' in text and '량' not in text):
        if not dose['total_dose']:
            dose['total_dose'].append([_x1 + x3, _x1 + diff + x3])
        else:
            dose['total_dose'][0][0] = min(dose['total_dose'][0][0], _x1 + x3)
            dose['total_dose'][0][1] = max(dose['total_dose'][0][1], _x1 + diff + x3 + 10)
        return True
    return False


def image_contrast(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    gray = Image.fromarray(gray)
    gray = gray.convert("RGB")
    gray = ImageEnhance.Contrast(gray).enhance(3)
    gray = np.array(gray)
    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2BGR)
    return gray


def extract_information(img, crafts, refine):
    res = {'drugs': []}
    (H, W) = img.shape[:2]
    boxes = text_detector.text_detect(crafts, refine, img)
    box = []

    for datas in boxes:
        data = [(int(datas[0][1]), int(datas[0][0])), (int(datas[2][1]), int(datas[2][0]))]
        box.append(data)

    box.sort()

    dose = {'single_dose': [], 'daily_dose': [], 'total_dose': []}

    # 1회, 1일, 총 복용량 위치 구하기
    for (y1, x1), (y3, x3) in box:
        crop = img[y1:y3, x1:x3]
        gray = cv2.pyrUp(crop)
        title = pt.image_to_string(gray, config='--psm 6 --oem 3 -c preserve_interword_spaces=1', lang='kor')
        title = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', title)
        if '처방 의약품의' in title or '의약품의' in title:
            diff = (y3 - y1)
            diff += diff//2
            croped = img[y1 - diff:y3 + diff, x3:W]
            dosebox = text_detector.text_detect(crafts, refine, croped)
            box2 = []
            for datas in dosebox:
                data = [(int(datas[0][0]), int(datas[0][1])), (int(datas[2][0]), int(datas[2][1]))]
                box2.append(data)
            box2.sort()
            for (_x1, _y1), (_x3, _y3) in box2:
                ncrop = croped[_y1:_y3, _x1:_x3]
                text = pt.image_to_string(ncrop, config='--psm 6 --oem 3 -c preserve_interword_spaces=1', lang='kor')
                if insert_dose(dose, text, _x1, _x3, x3): continue
                ncrop = image_contrast(ncrop)
                text = pt.image_to_string(ncrop, config='--psm 6 --oem 3 -c preserve_interword_spaces=1', lang='kor')
                insert_dose(dose, text, _x1, _x3, x3)
            break
        gray = image_contrast(gray)
        title = pt.image_to_string(gray, config='--psm 6 --oem 3 -c preserve_interword_spaces=1', lang='kor')
        title = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', title)
        if '처방 의약품의' in title or '의약품의' in title:
            diff = (y3 - y1)
            diff += diff//2
            croped = img[y1 - diff:y3 + diff, x3:W]
            dosebox = text_detector.text_detect(crafts, refine, croped)
            box2 = []
            for datas in dosebox:
                data = [(int(datas[0][0]), int(datas[0][1])), (int(datas[2][0]), int(datas[2][1]))]
                box2.append(data)
            box2.sort()
            for (_x1, _y1), (_x3, _y3) in box2:
                ncrop = croped[_y1:_y3, _x1:_x3]
                text = pt.image_to_string(ncrop, config='--psm 6 --oem 3 -c preserve_interword_spaces=1', lang='kor')
                if insert_dose(dose, text, _x1, _x3, x3): continue
                ncrop = image_contrast(ncrop)
                text = pt.image_to_string(ncrop, config='--psm 6 --oem 3 -c preserve_interword_spaces=1', lang='kor')
                insert_dose(dose, text, _x1, _x3, x3)
            break

    if not dose['single_dose'] or not dose['daily_dose'] or not dose['total_dose']:
        (H, W) = img.shape[:2]
        if dose['single_dose']:
            x1, x3 = dose['single_dose'][0]
            if not dose['daily_dose'] and not dose['total_dose']:
                diff = x3 - x1
                diff = diff//4
                dose['daily_dose'].append([x3 + diff, x3 + diff*2 + diff//2])
                dose['total_dose'].append([x3+diff*2 + diff//2, min(x3 + diff*4,W)])
            elif dose['daily_dose'] and not dose['total_dose']:
                x1, x3 = dose['daily_dose'][0]
                diff = x3 - x1
                dose['total_dose'].append([x3 + diff//2, x3 + diff + diff//2])
            else:
                _x1, _x3 = dose['total_dose'][0]
                diff = _x3 - _x1
                dose['daily_dose'].append([x3 + diff//2, x3 + diff + diff//2])
        elif dose['daily_dose']:
            x1, x3 = dose['daily_dose'][0]
            if dose['total_dose']:
                diff = x3 - x1
                dose['single_dose'].append([x1 - diff//2 - diff, x1 - diff//2])
            else:
                diff = x3 - x1
                dose['single_dose'].append([x1 - diff//2 - diff, x1 - diff//2])
                dose['total_dose'].append([x3 + diff//2, x3 + diff//2 + diff])
        else:
            if dose['total_dose']:
                x1, x3 = dose['total_dose'][0]
                diff = x3 - x1
                dose['daily_dose'].append([x1 - diff - diff//2, x1 - diff//2])
                dose['single_dose'].append([x1 - diff*4, x1 - diff*2])

    # 각 박스를 보며 약품 코드의 박스인 경우 DB비교 후 결과에 저장
    for (y1, x1), (y3, x3) in box:
        crop = img[y1:y3, x1:x3]
        gray = cv2.pyrUp(crop)

        text = pt.image_to_string(gray, config='--psm 6 --oem 3 -c preserve_interword_spaces=1', lang='kor')
        text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', text)
        code = string_process(text)

        if code:
            drug = db.selectQuery(code)
            if drug and drug[0]:
                for obj in dose:
                    if not dose[obj]: continue
                    _x1, _x3 = dose[obj][0]
                    number = img[y1:y3, _x1:_x3]
                    if obj == 'total_dose':
                        number = img[y1-5:y3+5, _x1-5:_x3+5]
                    number = cv2.pyrUp(number)
                    number = cv2.cvtColor(number, cv2.COLOR_BGR2GRAY)
                    text = pt.image_to_string(number, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
                    if text and int(text) <= 30:
                        drug[0][obj] = text
                        cv2.imwrite('numbers/' + str(drug[0]['code']) + obj + '_res_' + text + '.jpg', number)
                    else:
                        number = cv2.cvtColor(number, cv2.COLOR_GRAY2BGR)
                        number = image_contrast(number)
                        text = pt.image_to_string(number,
                                                  config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
                        if text and int(text) <= 30:
                            drug[0][obj] = text
                            cv2.imwrite('numbers/' + str(drug[0]['code']) + obj + '_res_' + text + '.jpg', number)
                res['drugs'].append(drug[0])
                continue

        gray = image_contrast(crop)
        text = pt.image_to_string(gray, config='--psm 6 --oem 3 -c preserve_interword_spaces=1', lang='kor')
        text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', text)
        code = string_process(text)

        if code:
            drug = db.selectQuery(code)
            if not drug: continue
            if drug[0]:
                for obj in dose:
                    if not dose[obj]: continue
                    (_x1, _x3) = dose[obj][0]
                    number = img[y1:y3, _x1:_x3]
                    if obj == 'total_dose':
                        number = img[y1-5:y3+5, _x1-5:_x3+5]
                    number = cv2.pyrUp(number)
                    number = cv2.cvtColor(number, cv2.COLOR_BGR2GRAY)
                    text = pt.image_to_string(number, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
                    if text and int(text) <= 30:
                        drug[0][obj] = text
                        cv2.imwrite('numbers/' + str(drug[0]['code']) + obj + '_res_' + text + '.jpg', number)
                    else:
                        number = cv2.cvtColor(number, cv2.COLOR_GRAY2BGR)
                        number = image_contrast(number)
                        text = pt.image_to_string(number,
                                                  config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
                        if text and int(text) <= 30:
                            drug[0][obj] = text
                            cv2.imwrite('numbers/' + str(drug[0]['code']) + obj + '_res_' + text + '.jpg', number)
                res['drugs'].append(drug[0])
    return res


# 주어진 사진이 명확하게 처방전의 가장 바깥쪽 직선을 포함한 경우 Warping
def edge_detect(img):
    (H, W) = img.shape[:2]
    # 그레이스 스케일 변환 및 케니 엣지
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # 가우시안 블러로 노이즈 제거
    try:
        edged = cv2.Canny(gray, 75, 200)  # 케니 엣지로 경계 검출
        # 컨투어 찾기
        (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        if width > W * 0.5 and height > H * 0.5:
            # 변환 후 4개 좌표
            pts2 = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
            # 변환 행렬 계산
            mtrx = cv2.getPerspectiveTransform(pts1, pts2)
            # 원근 변환 적용
            result = cv2.warpPerspective(img, mtrx, (width, height))
            return result, True
        else:
            result = img
            return result, False
    except Exception:
        result = img
        return result, False


def image_warp(src_loc, crafts, refine):
    # 이미지 읽기
    img = cv2.imread(src_loc)
    img, rot = image_resize(img)
    img = hough_linep(img)
    if rot:
        res = extract_information(img, crafts, refine)
        if res['drugs']:
            return res
        img = rotate_180deg(img)
        res = extract_information(img, crafts, refine)
    else:
        res = extract_information(img, crafts, refine)
    del img
    return res


def accuracy_test(crafts, refine):
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
            ret = image_warp(file, crafts, refine)
        except Exception as e:
            print('error report :', e)
            continue
        total_document += 1
        cnt += len(result['drugs'])
        tempcnt = 0
        for obj in ret['drugs']:
            code = str(obj['code'])
            single = str(int(float(obj['single_dose'])))
            daily = str(obj['daily_dose'])
            total = str(obj['total_dose'])
            target = next((item for item in result['drugs'] if item['code'] == code), None)
            if not target: continue
            codecnt += 1
            tempcnt += 1
            if target['single_dose'] == single:
                singlecnt += 1
            if target['daily_dose'] == daily:
                dailycnt += 1
            if target['total_dose'] == total:
                totalcnt += 1
            print('Answer', code, single, daily, total)
            print('Recog', target['code'], target['single_dose'], target['daily_dose'], target['total_dose'])
        print(cnt, codecnt, singlecnt, dailycnt, totalcnt)
    print('Total Processed Document : ', total_document)
    print('Code Accuracy :', (codecnt / cnt) * 100)
    print('Single Dose Accuracy :', (singlecnt / cnt) * 100)
    print('Daily Dose Accuracy :', (dailycnt / cnt) * 100)
    print('Total Dose Accuracy :', (totalcnt / cnt) * 100)


if __name__ == '__main__':
    crafts, refine = load_craft()
    accuracy_test(crafts, refine)
