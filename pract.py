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
import datetime as pydatetime

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
    if width * 3 == height * 4 or height*16 == width*9:
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


# 정규표현식을 통해 9글자 코드를 추출, 숫자가 한글자도 없는 경우 약물 코드로 판단하지 않음
def string_process(text):
    codes = re.findall('[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+[0-9]+', text)
    if codes:
        try:
            code = int(codes[0])
            if code > 999999999:
                code /= 10
            code = int(code)
            str(code)
            return code
        except Exception as e:
            return None
    return None


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


def image_contrast(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    gray = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # gray = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
    # gray = Image.fromarray(gray)
    # gray = gray.convert("RGB")
    # gray = ImageEnhance.Contrast(gray).enhance(3)
    # gray = np.array(gray)
    # gray = cv2.cvtColor(gray, cv2.COLOR_RGB2BGR)
    return gray


def replace_chars(text):
    list_of_numbers = re.findall(r'\d+', text)
    return list_of_numbers


def extract_information(img, crafts, refine):
    res = {'drugs':[]}
    (H, W) = img.shape[:2]
    boxes = text_detector.text_detect(crafts, refine, img)
    box = []
    for datas in boxes:
        data = [(int(datas[0][0]), int(datas[0][1])), (int(datas[2][0]), int(datas[2][1]))]
        box.append(data)
    box.sort()
    for (x1, y1), (x3, y3) in box:
        crop = img[y1:y3, x1:x3]
        crop = cv2.pyrUp(crop)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        gray = Image.fromarray(gray)
        gray = gray.convert("RGB")
        gray = ImageEnhance.Contrast(gray).enhance(3)
        gray = np.array(gray)
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2BGR)
        text = pt.image_to_string(gray, config='--psm 6 --oem 3 -c preserve_interword_spaces=1', lang='kor')
        code = string_process(text)
        if code:
            print(code)
            drug = db.selectQuery(code)
            if not drug: continue
            if drug[0]:
                res['drugs'].append(drug[0])
                # crop = img[y1:y3, x3:W]
                # crop = cv2.pyrUp(crop)
                # gray = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                # gray = Image.fromarray(gray)
                # gray = gray.convert("RGB")
                # gray = ImageEnhance.Contrast(gray).enhance(3)
                # gray = np.array(gray)
                # gray = cv2.cvtColor(gray, cv2.COLOR_RGB2BGR)
                # text = pt.image_to_string(gray, config='--psm 6 --oem 3 -c preserve_interword_spaces=1', lang='kor')
                # text = replace_chars(text)
                # print('insert complte!')
                # print(drug[0]['code'])
                # print(text)
                # print_img(gray)

    return res


def image_warp(src_loc, crafts, refine):
    # 이미지 읽기
    img = cv2.imread(src_loc)
    img, rot = image_resize(img)
    img = hough_linep(img)
    img = image_contrast(img)
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

def process(crafts, refine):
    files = glob.glob('image/temp/*.jpg')
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
    crafts, refine = load_craft()
    process(crafts, refine)
