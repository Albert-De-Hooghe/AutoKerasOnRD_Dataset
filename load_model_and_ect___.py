

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


model = keras.models.load_model('./image_classifier_bis/best_model/', custom_objects= ak.CUSTOM_OBJECTS)


plot_model(model, to_file='model.png')
