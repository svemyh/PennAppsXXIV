import os
from flask import Flask, request, render_template, send_from_directory
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import json

from utils import dummy_predict, average_pixel_value

app = Flask(__name__)

# Set the upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained InceptionV3 model
#model = InceptionV3(weights='imagenet')

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        
        # Check if a file is uploaded
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        # Check if the file has an allowed extension
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Load and preprocess the image for the model
            #img = image.load_img(filename, target_size=(299, 299))
            #img = image.img_to_array(img)
            #img = preprocess_input(img)
            #img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
            


            #result_json = dummy_predict(filename)
            #result_json = dummy_predict('flaskapp/static/uploads/upenn_2.jpg')
            # Parse the JSON string back into a Python dictionary
            #result_dict = json.loads(result_json)

            # Access and store the average_pixel_value
            #dummy_prediction_value = result_dict["average_pixel_value"]

            img_path = 'static/uploads/tajmahal_1.jpg'
            img_path = 'static/uploads/captured.jpg'

            dummy_prediction_value = average_pixel_value(img_path)

            result = ["Hello", str(dummy_prediction_value), "World!"]

            return render_template('index.html', message='File successfully uploaded and processed', result=result, img_path=img_path)

        else:
            return render_template('index.html', message='Invalid file extension')

    return render_template('index.html', message='Upload an image')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=False)
