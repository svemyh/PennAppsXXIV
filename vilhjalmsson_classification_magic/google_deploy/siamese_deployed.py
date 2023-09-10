import numpy as np
from keras.models import load_model
import keras.utils as image
from keras.optimizers import Adam
from keras import backend as K
from flask import Flask, request, jsonify
import google.cloud.storage

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
imgPath1 = 'image1.jpg'
imgPath2 = 'image2.jpg'
source1 = "path/to/image1.jpg"
source2 = "path/to/image2.jpg"

def download_blob(destination_file_name, source_blob_name, bucket_name="my-bucket"):
    """Downloads a blob from the bucket."""
    storage_client = google.cloud.storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

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

@app.route('/similarity', methods=['POST'])
def similarity():
    try:
        similarity = calculate_similarity(imgPath1, imgPath2, siamese_network)
        data = {"similarity": 1-similarity, "status": "success"}
        return jsonify(data)
    except Exception as e:
        data = {"status": "error", "message": str(e)}
        return jsonify(data)



from flask import Flask, flash, request, redirect, url_for, render_template
import os
from werkzeug.utils import secure_filename
import urllib.request

app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'files[]' not in request.files:
		flash('No file part')
		return redirect(request.url)
	files = request.files.getlist('files[]')
	file_names = []
	for file in files:
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file_names.append(filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		else:
			flash('Allowed image types are -> png, jpg, jpeg, gif')
			return redirect(request.url)

	return render_template('upload.html', filenames=file_names)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)