import flask
import werkzeug
import time
import image_proc as imp
import os
from flask import request, jsonify, make_response
import json
import chat
from OpenSSL import SSL

app = flask.Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/', methods = ['GET' , 'POST'])
def hello():
    return 'Hello Main!'

@app.route('/webhook', methods = ['POST'])
def webhook():
    message = request.form['message']
    return chat.get_answer(message,"1234")

@app.route('/image', methods = ['POST'])
def handle_request():
    files_ids = list(flask.request.files)
    print("\nNumber of Received Images : ", len(files_ids))
    for file_id in files_ids:
        print("\nRead Image ...")
        imagefile = flask.request.files[file_id]
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        print("Image Filename : " + imagefile.filename)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        imagefile.save(timestr+'_'+filename)
        try:
            ret = imp.image_warp(timestr+'_'+filename)
        except Exception as ex :
            print('에러가 발생했습니다.', ex)
            return 'Image Processing Failed!'
        os.remove(timestr+'_'+filename)
    print("Send Complete!\n")
    return json.dumps(ret, ensure_ascii=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
