import flask
import werkzeug
import time
import image_proc as imp
import os
from flask import request
import json
import chat
import cv2

app = flask.Flask(__name__)
app.config['JSON_AS_ASCII'] = False
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet('frozen_east_text_detection.pb')


@app.route('/', methods=['GET', 'POST'])
def hello():
    return 'Hello Main!'


@app.route('/webhook', methods=['POST'])
def webhook():
    message = request.form['message']
    user_name = request.form['username']
    return chat.detect_intent_texts(message, user_name)


@app.route('/image', methods=['POST'])
def handle_request():
    files_ids = list(flask.request.files)
    print("\nNumber of Received Images : ", len(files_ids))
    for file_id in files_ids:
        print("\nRead Image ...")
        imagefile = flask.request.files[file_id]
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        print("Image Filename : " + imagefile.filename)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        imagefile.save(timestr + '_' + filename)
        try:
            ret = imp.image_warp(timestr + '_' + filename, net)
        except Exception as ex:
            print('에러가 발생했습니다.', ex)
            return 'Image Processing Failed!'
        try:
            os.remove(timestr + '_' + filename)
        except Exception as ex:
            print('파일 관리에 실패했습니다', ex)
    print("Send Complete!\n")
    return json.dumps(ret, ensure_ascii=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
