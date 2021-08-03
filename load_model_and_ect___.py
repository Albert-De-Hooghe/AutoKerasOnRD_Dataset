

# from keras.models import load_model
# from    keras.utils.vis_utils    import plot_model
# import autokeras as ak
# MODEL_DIR = './image_classifier/best_model/'
# MODEL_PNG = './model_PNG_chouette.png'
#
# ##
# #
# # export_keras_model(MODEL_DIR) the example available here: https://www.programmersought.com/article/4500629357/
# loaded_model = load_model("model_autokeras", custom_objects=ak.CUSTOM_OBJECTS)
# model = load_model('./image_classifier/best_model')
#
# plot_model(model, to_file=MODEL_PNG,show_shapes=True)

import autokeras as ak
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from readRD_dataset import RD_Dataset_train_5_classes, RD_Dataset_valid_5_classes
import numpy as np
import cv2

def plot_the_model(model):

    ### plot the model (obtaining a png)

    plot_model(model, to_file='model.png')

def make_a_prediction(model, batch_of_image):

    ### making a prediction

    return model(batch_of_image)

def read_one_image_from_the_path(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, [1, 224, 224, 3])
    return img

if __name__ == "__main__":

    ### loadding the model

    model = keras.models.load_model('./image_classifier/best_model/', custom_objects=ak.CUSTOM_OBJECTS)
    taille = "mini"


    ## load data

    # (x_train, y_train), (x_test, y_test) = mnist.load_data() # exemple de base
    x_valid_RD, y_valid_RD = RD_Dataset_valid_5_classes(taille)
    x_train_RD, y_train_RD = RD_Dataset_train_5_classes(taille)

    path_of_image_folder = "/home/albert/ClusterOvh/darts_data_nni/preprocessedTrain224/"
    name_of_the_image = "23269_left.png"

    path_of_one_image_for_test = path_of_image_folder+name_of_the_image
    image = read_one_image_from_the_path(path_of_one_image_for_test)
    print(model)
    print("prediction is:", make_a_prediction(model, image))
