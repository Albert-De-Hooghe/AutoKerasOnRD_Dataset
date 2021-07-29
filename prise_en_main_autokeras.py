

#!pip install autokeras

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import autokeras as ak
from tensorflow.python.client import device_lib
from readRD_dataset import RD_Dataset_train_5_classes, RD_Dataset_valid_5_classes
device_name = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']

MODEL_DIR = "./saving_models"

print("device name is :", device_name)

taille = "mini"


# (x_train, y_train), (x_test, y_test) = mnist.load_data() # exemple de base
x_valid_RD, y_valid_RD = RD_Dataset_valid_5_classes(taille)
x_train_RD, y_train_RD = RD_Dataset_train_5_classes(taille)
# print("size is:", x_train_RD.shape)
# print("size is:", y_train_RD.shape)
#
# print(y_train_RD)
# print(x_train.shape)
#
# print(x_train_RD[0])
# print(y_train.shape)


#print(x_test_RD[0])
#print(x_test[0])

clf = ak.ImageClassifier(num_classes=5, overwrite=True, max_trials=1, tuner='bayesian', project_name='image_classifier_bis')

clf.fit(x_train_RD, y_train_RD, epochs=10, validation_data=(x_valid_RD, y_valid_RD))

model = clf.export_model()


try:
    model.save("model_autokeras", save_format="tf")
except Exception:
    model.save("model_autokeras.h5")


#predicted_y = clf.predict(x_test_RD)
#print(predicted_y)

#print(clf.evaluate(x_test_RD,y_test_RD))


