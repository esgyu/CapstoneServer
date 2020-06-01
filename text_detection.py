from imutils.object_detection import non_max_suppression
import numpy as np
import time
import pytesseract as pt
import re
from tqdm import tqdm
import db_connect as db
import cv2

def stringProcess(text):
	# 9글자로 이루어진 의약품 코드 추출
	char = re.findall('[가-힣]+', text)
	if char:
		return None
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

def printimg(img):
    cv2.imshow('aaa', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def text_roi_extension(image, startX, endX, startY, endY, W, H):
	startX = max(startX - 5, 0)
	endX = min(endX + 5, W)
	startY = max(startY - 5, 0)
	endY = min(endY + 10, H)

	img = image[startY:endY, startX:endX]
	img = cv2.pyrUp(img)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
	text = pt.image_to_string(gray, config='--psm 6', lang='kor')
	res = stringProcess(text)

	if not res:
		return None
	if res != 'maybe':
		result = db.selectQuery(res)
		if result:
			return result

	# x축 증가
	for i in range(10, 100, 10):
		if endX+i >= W:
			break
		img = cv2.pyrUp(image[startY+i//50:endY+i//50, startX:endX+i])
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
		text = pt.image_to_string(gray, config='--psm 6', lang='kor')
		res = stringProcess(text)
		if res=='maybe':
			continue
		if res:
			result = db.selectQuery(res)
			if result:
				return result
		else:
			break

	# x축 감소
	for i in range(10, 100, 10):
		if startX-i<0 :
			break
		img = cv2.pyrUp(image[startY+i//50:endY+i//50, startX-i: endX])
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
		text = pt.image_to_string(gray, config='--psm 6', lang='kor')
		res = stringProcess(text)
		if res == 'maybe':
			continue
		if res:
			result = db.selectQuery(res)
			if result:
				return result
		else:
			break
	# 4방향 동시 증가
	for i in range(10, 100, 10):
		img = cv2.pyrUp(image[max(int(startY - i//30), 0):min(int(endY + i//30), H), max(startX - i, 0):min(endX+i,W)])
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		text = pt.image_to_string(gray, config='--psm 6', lang='kor')
		res = stringProcess(text)
		if res == 'maybe':
			continue
		if res:
			result = db.selectQuery(res)
			if result:
				return result
		else:
			break
	return None

def text_detect(image, net, layerNames):
	(H, W) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),(123.68, 116.78, 103.94), swapRB=True, crop=False)

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

			# add the bounding box coordinates and probability score to
			# our respective lists
			if (endX - startX) < 40:
				continue
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	boxes = non_max_suppression(np.array(rects), probs=confidences)
	ret = {'drugs': []}

	for (startX, startY, endX, endY) in tqdm(boxes):
		try:
			res = text_roi_extension(image, startX, endX, startY, endY, W, H)
			if res:
				ret['drugs'].append(res[0])
			# draw the bounding box on the image
		except Exception as ex:
			print('error report : ' , ex)

	return ret

if __name__ == '__main__' :
	image = cv2.imread('KakaoTalk_20200508_233954026.jpg')
	text_detect(image)