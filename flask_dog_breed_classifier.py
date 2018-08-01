#open http://127.0.0.1:5000/ on your browser to see result

from flask import Flask, flash, request, redirect, url_for
from flask import send_file
from werkzeug.utils import secure_filename
from flask import render_template
import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras.applications import xception
from keras.applications import inception_v3
from os.path import join
import pickle
import matplotlib.pyplot as plt
import io
import os

#extract bottleneck
inception_bottleneck = inception_v3.InceptionV3(weights='./models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, pooling='avg')
inception_bottleneck._make_predict_function()
xception_bottleneck = xception.Xception(weights='./models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, pooling='avg')
xception_bottleneck._make_predict_function()
#load model
logreg = pickle.load(open('./models/logreg_model.sav', 'rb'))

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg']) #permitted file extensions for user upload

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(join(app.config['UPLOAD_FOLDER'], filename)) #save user input
            return render_template('index.html', filename=filename) #display result in html
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload your dog picture to predict its breed</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
    
@app.route('/uploaded_file/<filename>')
def uploaded_file(filename):
     #import labels
     NUM_CLASSES = 120
     labels = pd.read_csv('./labels.csv')
     selected_breed_list = list(labels.groupby('breed').count().sort_values(by='id', ascending=False).head(NUM_CLASSES).index)

     # to make new predictions, run code below this line
     img_path = join(app.config['UPLOAD_FOLDER'], filename) 
     img = image.load_img(img_path, target_size=(299, 299))
     img = image.img_to_array(img)
     img_prep = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))
     imgX = xception_bottleneck.predict(img_prep, batch_size=32, verbose=1)
     imgI  = inception_bottleneck.predict(img_prep, batch_size=32, verbose=1)
     img_stack = np.hstack([imgX, imgI])
     prediction = logreg.predict(img_stack)
    
     #plot image and prediction
     fig, ax = plt.subplots(figsize=(5,5))
     ax.imshow(img / 255.)
     breed = selected_breed_list[int(prediction)]
     ax.text(10, 250, 'Prediction: %s' % breed, color='k', backgroundcolor='g', alpha=0.8)
     ax.axis('off')
     output = io.BytesIO()
     fig.savefig(output)
     output.seek(0)
     if os.path.exists(img_path):
         os.remove(img_path) #delete user input
     return send_file(output, mimetype='image/png') #return the output image to be called in html