import tensorflow as tf  
from tensorflow.keras import layers, models  
  
def build_deep_metric_model(input_shape):  
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)  
    base_model.trainable = False  
    flatten = layers.Flatten()(base_model.output)  
    dense = layers.Dense(256, activation='relu')(flatten)  
    model = models.Model(base_model.input, dense)  
      
    return model  
  
def contrastive_loss(y_true, y_pred, margin=1.0):  
    square_pred = tf.square(y_pred)  
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))  
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)  