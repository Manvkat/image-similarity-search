import tensorflow as tf  
from tensorflow.keras.applications import VGG16  
from tensorflow.keras.models import Model  
  
def build_cnn_feature_extractor(input_shape):  
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)  
    output = base_model.layers[-1].output  
    output = tf.keras.layers.GlobalAveragePooling2D()(output)  
    model = Model(inputs=base_model.input, outputs=output)  
      
    for layer in base_model.layers:  
        layer.trainable = False  
      
    return model  