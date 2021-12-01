import tensorflow as tf
import flask
import werkzeug
from tensorflow.keras import models
import numpy as np
import imageio
import os

app = flask.Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def welcome():
    return "Hello World"

@app.route('/predict/', methods = ['GET', 'POST'])
def handle_request():
    class_names = ['benign', 'malignant']
    imagefile = flask.request.files['image0']
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image file name : " + imagefile.filename)
    imagefile.save(filename)
    # img = imageio.imread(filename, pilmode="L")
    # if img.shape != (100, 100):
    #     return "Image size mismatch" + str(img.shape) + ". \nOnly (100, 100) is acceptable."
    # filepath = "ISIC-images/UDA-1/ISIC_0000016.jpg"
    # filepath = "benign/ISIC-images/UDA-1/ISIC_1193108.jpg"
    img = tf.keras.utils.load_img(
    	filename, target_size=(100, 100)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    loaded_model = models.load_model('benign_malignant_tuning10.h5')
    predictions = loaded_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_label = str("This image most likely belongs to {} with a {:.2f} percent confidence."
    	.format(class_names[np.argmax(score)], 100 * np.max(score)))
    # print(
    # 	"This image most likely belongs to {} with a {:.2f} percent confidence."
    # 	.format(class_names[np.argmax(score)], 100 * np.max(score))
    # )
    # img = imageio.imread(filename, pilmode="L")
    # if img.shape != (100, 100):
    #     return "Image size mismatch" + str(img.shape) + ". \nOnly (100, 100) is acceptable."
    # img = img.reshape(1, 100, 100, 3)
    # loaded_model = models.load_model('benign_malignant_tuning4.h5')
    # predicted_label = numpy.argmax(loaded_model.predict(numpy.array([img]))[0], axis=-1)
    # print(predicted_label)

    return str(predicted_label)

app.run(host="0.0.0.0", port=os.environ.get('PORT', 5000), debug=True)