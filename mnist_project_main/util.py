from keras.models import model_from_json
import numpy
import os


def save_model(filename, model):
	filejson = "models/"+filename + ".json"
	fileh5 = "models/" +filename + ".h5"
	# serialize model to JSON
	model_json = model.to_json()
	with open(filejson, "w") as json_file:
	    json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights(fileh5)
	print("Model saved to disk")

def load_model(filename):
	filejson = "models/"+filename + ".json"
	fileh5 = "models/" +filename + ".h5"
	json_file = open(filejson, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(fileh5)
	print("Loaded model from disk")

	return loaded_model