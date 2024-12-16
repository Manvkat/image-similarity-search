import tensorflow as tf  
from tensorflow.keras import layers, models  
  
def build_triplet_model(input_shape):  
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)  
    base_model.trainable = False  
    flatten = layers.Flatten()(base_model.output)  
    dense = layers.Dense(256, activation='relu')(flatten)  
    model = models.Model(base_model.input, dense)  
      
    return model  
  
def triplet_loss(y_true, y_pred, margin=1.0):  
    anchor, positive, negative = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]  
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)  
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)  
    loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)  
    return tf.reduce_mean(loss)  