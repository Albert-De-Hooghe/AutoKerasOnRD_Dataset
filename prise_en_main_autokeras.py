

#!pip install autokeras

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import autokeras as ak

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(y_train[:3])

print(y_train[1:15])

clf = ak.ImageClassifier(overwrite = True, max_trials=1)

clf.fit(x_train, y_train, epochs=10)

predicted_y = clf.predict(test)
print(predicted_y)

print(clf.evaluate(x_test, y_test))## j'ai stoppé car ça fait trop chauffer le mon zenbook.


