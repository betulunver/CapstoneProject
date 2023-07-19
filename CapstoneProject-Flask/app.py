import numpy as np
from flask import Flask, request, jsonify, render_template
import os

from PIL import Image
from keras.utils import load_img, img_to_array
from keras.models import load_model 
from math import expm1

app = Flask(__name__)

image_folder = os.path.join('static', 'images')
meta=os.path.join('static', 'Meta')
app.config["UPLOAD_FOLDER"] = image_folder
app.config["META"]=meta

model = load_model('model/trafficsign.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

@app.route('/', methods=['POST'])
def predict():
  # predicting images
  imagefile = request.files['file']
  if(imagefile):
        image_path = './static/images/' + imagefile.filename 
        imagefile.save(image_path)

        img = load_img(image_path, target_size=(30,30))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)
        maxPrediction = classes.argmax() # en çok benzeyen fotoğrafın index'ini aldım.(classes txt'de sıralı olarak yer alıyor)

        pic = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
        pred_pic=os.path.join(app.config['META'], str(int(maxPrediction))+".png") # tahmin edilen fotoğrafı göstermek için

        print(pred_pic,"----",maxPrediction)
        return render_template('index.html', user_image=pic, pred_image=pred_pic)
  else:
        return render_template('index.html',prediction_text='Please select file !')
if __name__ == "__main__":
    app.run(debug=True)