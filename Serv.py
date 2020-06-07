import json
import os
import time
from collections import OrderedDict

import cv2
import flask
import torch
import werkzeug
from flask import request

import chat
from crafts import craft
from crafts import text_detector
from crafts import refinenet
import image_proc as imp

app = flask.Flask(__name__)
app.config['JSON_AS_ASCII'] = False

net = None
refine_net = None
crafts = None

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
        filename = werkzeug.utils.secure_filename(imagefile.filename) + '_' + time.strftime("%Y%m%d-%H%M%S") + '.jpg'
        filename = os.path.join('image', filename)
        imagefile.save(filename)
        try:
            ret = imp.image_warp(filename, net, crafts, refine_net)
            #ret = text_detector.text_detect(net, refine_net, filename)
        except Exception as ex:
            print('에러가 발생했습니다.', ex)
            return 'Image Processing Failed!'

        try:
            os.remove(filename)
        except Exception as ex:
            print('파일 관리에 실패했습니다', ex)

    print("Send Complete!\n")
    return json.dumps(ret, ensure_ascii=False)


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


def load_east():
    print("[INFO] loading EAST text detector...")
    return cv2.dnn.readNet('frozen_east_text_detection.pb')


if __name__ == "__main__":
    crafts, refine_net = load_craft()
    net = load_east()
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
