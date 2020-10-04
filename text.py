# from __future__ import division, print_function
# import sys
# import os
# import glob
# import re
# import numpy as np

# import keras
# from keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from keras.models import load_model
# from keras.preprocessing import image

from flask import Flask, render_template, url_for 
# from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer


app= Flask(__name__)

# MODEL_PATH = 'models/nsac_model.h5'
# model = load_model(MODEL_PATH)
# model._make_predict_function()
# print('Model loaded. Check http://127.0.0.1:5000/')


# def model_predict(img_path, model):
#     img = image.load_img(img_path, target_size=(224, 224))

#     # Preprocessing the image
#     x = image.img_to_array(img)
#     # x = np.true_divide(x, 255)
#     x = np.expand_dims(x, axis=0)

#     # Be careful how your trained model deals with the input
#     # otherwise, it won't make correct prediction!
#     x = preprocess_input(x, mode='caffe')

#     preds = model.predict(x)
#     return preds
# UPLOAD_FOLDER = './uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 


@app.route('/')
def index():
    return render_template('index.html')

# def upload_file():
#    if request.method =='POST':
#         file = request.files['file']
#         if file:
#              filename = secure_filename(file.filename)
#              file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
#             return index()
#     return render_template('file_upload.html')

# @app.route("/file_upload", methods=["POST", "GET"])
# def file_upload():
#     if request.method == "POST":
#         f = request.files['file']
#         # Save the file to ./uploads
#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(
#             basepath, 'uploads', secure_filename(f.filename))
#         f.save(file_path)

#         preds = model_predict(file_path, model)
#         pred_class = decode_predictions(preds, top=1)
#         result = str(pred_class[0][0][1])              
#         return result

#     return None
@app.route('/weather')
def weather():
    return render_template('weather.html')

if __name__ == "__main__":
    app.run(debug=True)