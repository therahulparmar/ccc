from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import pickle
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#import WSGIServer
#from gevent.pywsgi import WSGIServer
from PIL import Image
import psycopg2 #pip install psycopg2 
import psycopg2.extras
# import torchvision
# import torch
import numpy as np
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.applications import VGG16
#from tensorflow.keras.layers import AveragePooling2D
#from tensorflow.keras.layers import Dropout
#from tensorflow.keras.layers import Flatten
#from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.utils import to_categorical
#from sklearn.preprocessing import MultiLabelBinarizer
# import matplotlib.pyplot as plt
# import seaborn as sns
import cv2
import os
MODEL_PATH = 'models/resnet_chest.h5'
from tensorflow.keras.models import load_model
# Recreate the exact same model, including its weights and the optimizer
new_model = load_model('models/resnet_chest.h5')

# def test_rx_image_for_Covid19(file_path):
#     img = cv2.imread(file_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (224, 224))
#     img = np.expand_dims(img, axis=0)

#     img = np.array(img) / 255.0

#     pred = new_model.predict(img)
#     pred_neg = round(pred[0][1] * 100)
#     pred_pos = round(pred[0][0] * 100)

def test_rx_image_for_Covid19(imagePath):
    img = cv2.imread(imagePath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)

    img = np.array(img) / 255.0

    pred = new_model.predict(img)

    return pred

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    # print(np.argmax(pred, axis=1))

    #print('\n X-Ray Covid-19 Detection using AI - MJRovai')
    #print('    [WARNING] - Only for didactic purposes')
#     if np.argmax(pred, axis=1)[0] == 1:
#         plt.title(
#             '\nPrediction: [NEGATIVE] with prob: {}% \nNo Covid-19\n'.format(
#                 pred_neg),
#             fontsize=12)
#     else:
#         plt.title(
#             '\nPrediction: [POSITIVE] with prob: {}% \nPneumonia by Covid-19 Detected\n'
#             .format(pred_pos),
#             fontsize=12)

#     #img_out = plt.imread(file_path)
#     #plt.imshow(img_out)
#     #plt.savefig('../Image_Prediction/Image_Prediction.png')
#     return pred_pos


print('Model loaded. Check http://127.0.0.1:5000/')
app = Flask(__name__)
     
app.secret_key = "cairocoders-ednalan"
     
DB_HOST = "ec2-52-22-81-147.compute-1.amazonaws.com"
DB_NAME = "dc4dvtu6fb2ha2"
DB_USER = "udtsfbvuwfhjee"
DB_PASS = "998428904a9c98bfd3a7e20f0b338eb6b987dde755f49c9c8d8807ecb0524e78"
     
conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)
print("Database opened successfully")
UPLOAD_FOLDER = 'static/uploads/'
print("upload successfull")
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
  
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
  
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
      
  
@app.route('/')
def home():
    return render_template('index.html')
  
@app.route('/', methods=['POST'])
def upload_image():
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    print("cursor connection success")
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
        print("121")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        print("file name received")
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print("126")
        #print('upload_image filename: ' + filename)
#         file_path = ('static/uploads/Normal-15.png')
        pred = test_rx_image_for_Covid19('static/uploads/'+filename)
        print("100% covid")
        #print('Probabilities:', probabilities)
        #print('Predicted class index:', predicted_class_index)
        #print('Predicted class name:', predicted_class_name)
        #probabilities.values.astype(int)
        #predicted_class = predicted_class_name.tolist()
        #prob = float(np.mean(probabilities))
        name=request.form["name"]
        age=request.form["age"]
        city=request.form["city"]
        state=request.form["state"]
        pincode=request.form["pincode"]
        mobile=request.form["mobile"]
        gender=request.form["gender"]
        bloodgroup=request.form["bloodgroup"]
        print("all data received")
        prob = pred[0,1]*100
        prob_n = 100-prob
        if (pred[0,1] <= 0.5):
            pre_c_name = 'Covid Positive'
            prob_c = 100-prob
        else:
            pre_c_name = 'Covid Negative'
            prob_c = pred[0,1]*100
        cursor.execute("INSERT INTO db_cc_Photo (img , name , age, city, state, pincode, mobile, gender, bloodgroup,predicted_class_name, probabilities ) VALUES (%s ,%s ,%s ,%s ,%s ,%s ,%s ,%s ,%s ,%s , %s)", (filename, name , age, city, state, pincode, mobile, gender, bloodgroup, pre_c_name,prob_c) )
        conn.commit() 
        #file_path = file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Make prediction
        
        #display(image_path)
        if (pred[0,1] <= 0.5):

            return render_template('index.html', prob_text='Patient is Covid-19 Positive And Probability is : {}'.format( prob_n) , filename=filename) 
        else:
            return render_template('index.html', prob_text='Covid-19 Negative And Probability is : {}'.format(prob ) , filename=filename)
 
        flash('Image successfully uploaded and displayed below')
        #return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)




@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
