import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from flask import Flask, render_template,request
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np

model = load_model('covidmodel.h5')
app = Flask(__name__)

@app.route('/')
def hello():
	return render_template("index.html")

@app.route('/index.html')
def index():
	return render_template("index.html")
@app.route('/faq.html')
def faq():
	return render_template("faq.html")
@app.route('/guide.html')
def guide():
	return render_template("guide.html")

@app.route('/pred', methods= ['POST'])
def pred():
	path=""
	if request.method == 'POST':
		f = request.files["userfile"]
		path = "./static/{}".format(f.filename)
		f.save(path)
		img = image.load_img(path,target_size=(224,224))
		test = image.img_to_array(img)
		x = np.expand_dims(test, axis=0)
		img_data = preprocess_input(x)
		classes = model.predict(img_data)
		if(classes[0][0]<=0 and classes[0][1]>0):

    			result = "Normal" 
		else:
    			result = "COVID-19 Positive"
		result_dict = {
		'image': path,
		'prediction': result
		}
		

	return render_template("index.html", result = result_dict)


if __name__ == '__main__':
	app.run(threaded = False)
