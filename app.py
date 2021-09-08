from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import PIL
from PIL import Image


# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

MODEL_PATH= r'C:\Users\Rama Sundar\Desktop\hellodoc\models\mobilenet.h5'

model=load_model(MODEL_PATH)
model.make_predict_function()

UPLOAD_FOLDER="/Desktop/hellodoc/static/"


def model_predict(img_path,model):


    img = image.load_img(img_path, target_size=(128,128))
    im=image.img_to_array(img)
    io=np.expand_dims(im,axis=0)
    i=io.astype('float32')/255
    preds = model.predict(i)
    return preds




@app.route('/',methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        image_file=request.files["image"]
        basepath=os.path.dirname(__file__)

        if image_file:
            image_location=os.path.join(basepath,'uploads', secure_filename(image_file.filename))

            image_file.save(image_location)
            preds=model_predict(image_location,model)

            b=preds[0][0]*100
            sk=preds[0][1]*100
            sl=preds[0][2]*100
            scc=preds[0][3]*100
            m=preds[0][4]*100
            v1=float("{:.2f}".format(b))
            v2=float("{:.2f}".format(sk))
            v3=float("{:.2f}".format(sl))
            v4=float("{:.2f}".format(scc))
            v5=float("{:.2f}".format(m))

            return render_template("index.html",b=v1,sk=v2,sl=v3,scc=v4,m=v5)
    return render_template("index.html",b=0,sk=0,sl=0,scc=0,m=0)

if __name__ == '__main__':
    app.run(debug=True)
