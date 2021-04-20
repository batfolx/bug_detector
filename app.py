from flask import Flask, request, Response, make_response
from detect_images import detect_image
from PIL import Image
import numpy as np
import os

os.environ.get('MODEL_PORT')
app = Flask(__name__)


@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return Response("{'error': 'No file specified'}", status=201, mimetype='application/json')
    file = request.files['file']
    image_np = np.ndarray(file.read())
    image_np_labels = detect_image(image_np)
    response = make_response(Image.fromarray(image_np_labels).tobytes())
    response.headers.set("Content-Type", "image/jpeg")
    response.headers.set('Content-Disposition', 'attachment')
    return response

app.run(port=os.environ.get('MODEL_PORT'))






