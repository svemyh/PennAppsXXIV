import os
from flask import Flask, request, render_template, send_from_directory
import json

from utils import dummy_predict, average_pixel_value

app = Flask(__name__)

# Set the upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        img_path = 'static/uploads/captured.jpg'



        # INSERT MACHINE LEARNING PERDICTION HERE
        dummy_prediction_value = average_pixel_value(img_path)





        result = ["Hello", str(dummy_prediction_value), "World!"]

        return render_template('index.html', message='File successfully uploaded and processed', result=result, img_path=img_path)

    return render_template('index.html', message='Upload an image')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
