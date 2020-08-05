import base64

import cv2
import numpy as np
from PIL import Image
from flask import render_template, request, jsonify

from app import app
from model.neural_net import NeuralNetwork, alphabets_mapper


#
# app = Flask(__name__)


@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html', the_title='Character Recognition using CNN')


@app.route('/get_canvas_data', methods=['GET', 'POST'])
def process_image_data():
    jpgtxt = request.args.get('img', 0, type=str)
    data = jpgtxt.split(',')[-1]
    data = base64.b64decode(data.encode('ascii'))

    g = open("temp.jpg", "wb")
    g.write(data)
    g.close()
    img = Image.open('temp.jpg').convert('RGBA')
    im = np.array(img)
    im = im.reshape(im.shape[0], (im.shape[1]*im.shape[2]))

    resized = np.asarray(255.0 - (np.float32(cv2.resize(im, (20,20), interpolation=cv2.INTER_AREA).reshape(20, 20) * 255)))
    fn = lambda x: 0 if x < 200 else 255
    vfunc = np.vectorize(fn)
    grayed = vfunc(resized)
    model = NeuralNetwork().pretrained_model('model/conv_cnn_2.h5')
    predicted_image = np.argmax(model.predict(grayed.reshape(1, 20, 20, 1)))
    result = 'You entered: ' + alphabets_mapper[predicted_image]
    return jsonify(result=result)


# if __name__ == '__main__':
#     app.run(debug=True)
