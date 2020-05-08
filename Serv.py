import flask
import werkzeug
import time
import image_proc as imp
import os
from flask import request, render_template

app = flask.Flask(__name__)

@app.route('/')
def hello():
    return '<p>Hello Main!</p>'

@app.route('/string', methods = ['GET', 'POST'])
def print_result():
    result = request.form['message']
    print(result)
    return result

@app.route('/image', methods = ['GET', 'POST'])
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
            imp.image_warp(timestr+'_'+filename)
        except Exception as ex :
            print('에러가 발생했습니다.', ex)
            return 'Image Processing Failed!'
        os.remove(timestr+'_'+filename)
    print("\n")
    return "Image(s) Uploaded Successfully. Come Back Soon."

app.run(host="0.0.0.0", port=5000, debug=True)