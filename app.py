from flask import Flask, request, Response, send_file
from detect_images import detect_image
from PIL import Image
from threading import Thread
import numpy as np
import time
import os

app = Flask(__name__)


@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return Response("{'error': 'No file specified'}", status=201, mimetype='application/json')
    file = request.files['file']

    # convert uploaded file into PIL image to transform to NumPy array
    image = Image.open(file.stream)
    image_np_labels = detect_image(np.asarray(image).copy())
    filename = f'tmp/{file.filename}'

    # save image to file system to send back to client
    Image.fromarray(image_np_labels).save(filename)
    Thread(target=delete_file, args=(filename, ), daemon=True).start()
    return send_file(filename)


def delete_file(filename, timeout=10):
    time.sleep(timeout)
    if os.path.exists(filename):
        os.remove(filename)


@app.route('/')
def home():
    return send_file('index.html')


app.run(host='0.0.0.0', port=os.environ.get('MODEL_PORT'))

