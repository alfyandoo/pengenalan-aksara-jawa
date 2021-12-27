from flask import Flask, request,render_template
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import re
import base64

app = Flask(__name__)

loaded_model = load_model("models/model.h5")

# decoding an image from base64 into raw representation
def convertImage(imgData1):
    imgstr = re.search(r'base64,(.*)', str(imgData1)).group(1)
    with open('image_user_converted.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))

# load image
def load_image(img_path):
  img = load_img(img_path, target_size=(100, 100), color_mode = "grayscale")
  img_tensor = img_to_array(img)
  img_tensor = np.expand_dims(img_tensor, axis=0)
  img_tensor /= 255.0
  return img_tensor

@app.route("/")
def index(name=None):
  return render_template('index.html', name=name)

@app.route("/predict/", methods=['POST', 'GET'])
def predict():
    class_name = [
        "ba", "ca", "da", "dha", "ga",
        "ha", "ja", "ka", "la", "ma",
        "na", "nga", "nya", "pa", "ra",
        "sa", "ta", "tha", "wa", "ya"
    ]

    img_user = request.get_data()
    convertImage(img_user)
    img = load_image('image_user_converted.png')
    pred = loaded_model.predict(img)
    print(pred)
    print(class_name[np.argmax(pred)])
    response = class_name[np.argmax(pred)]
    return str(response)

@app.route("/about")
def about():
    return render_template('about.html')

if __name__ == "__main__":
  app.run(debug=True)