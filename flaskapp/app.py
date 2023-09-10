import os
from flask import Flask, request, render_template, send_from_directory

from utils import average_pixel_value



import numpy as np
from keras.models import load_model
import keras.utils as image
from keras.optimizers import Adam
from keras import backend as K
from flask import Flask, request

def triplet_loss(inputs, alpha=0.2):
    anchor, positive, negative = inputs
    pos_dist = K.sum(K.square(anchor - positive), axis=-1)
    neg_dist = K.sum(K.square(anchor - negative), axis=-1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.mean(K.maximum(basic_loss, 0.0))
    return loss

def identity_loss(y_true, y_pred):
    return y_pred

custom_objects = {'triplet_loss': triplet_loss, 'identity_loss': identity_loss}

# Load the model without compiling
#siamese_network = load_model('siamese_network.keras', custom_objects=custom_objects, compile=False, safe_mode=False)
siamese_network = load_model('siamese_network.h5', custom_objects=custom_objects, compile=False, safe_mode=False)
# Compile the model manually
siamese_network.compile(optimizer=Adam(), loss=identity_loss)



imgPath1 = 'static/uploads/hus1.jpg'
imgPath2 = 'static/uploads/captured.jpg'


def preprocess_image(image_path, target_size=(224, 224)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def calculate_similarity(model, image1_path=imgPath1, image2_path=imgPath2):
    # Download the images from Google Cloud Storage
    #download_blob(imgPath1, source1)
    #download_blob(imgPath2, source2)
    # Preprocess the images
    img1 = preprocess_image(image1_path)
    img2 = preprocess_image(image2_path)

    # Extract the base network from the Siamese network
    base_network = model.layers[-3]

    # Extract embeddings for the two images
    embedding1 = base_network.predict(img1)
    embedding2 = base_network.predict(img2)

    # Compute the L2 distance between the embeddings
    distance = np.linalg.norm(embedding1 - embedding2)

    return distance





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

        file = request.files['file']
        
        # Check if the file is empty
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'captured.jpg')
        file.save(filename)


        img_path = 'static/uploads/captured.jpg'



        smiliarity_value = calculate_similarity(siamese_network, imgPath1, imgPath2)



        result = [str(smiliarity_value)]

        return render_template('index.html', message='File successfully uploaded and processed', result=result, img_path=img_path)

    return render_template('index.html', message='Upload an image')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" in request.files:
        image = request.files["image"]
        # Save the image to a desired location or process it as needed.
        # Example: image.save("uploads/filename.jpg")

        # Return a response to the client
        return "Image uploaded successfully"
    return "No image provided", 400


if __name__ == '__main__':
    app.run(debug=True)
