import tensorflow as tf  
from tensorflow.keras import layers, models  
  
def build_autoencoder(input_shape):  
    encoder = models.Sequential([  
        layers.Input(shape=input_shape),  
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),  
        layers.MaxPooling2D((2, 2), padding='same'),  
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),  
        layers.MaxPooling2D((2, 2), padding='same'),  
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),  
        layers.MaxPooling2D((2, 2), padding='same'),  
        layers.Flatten(),  
        layers.Dense(256, activation='relu')  
    ])  
      
    decoder = models.Sequential([  
        layers.Input(shape=(256,)),  
        layers.Dense(4 * 4 * 128, activation='relu'),  
        layers.Reshape((4, 4, 128)),  
        layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same'),  
        layers.UpSampling2D((2, 2)),  
        layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),  
        layers.UpSampling2D((2, 2)),  
        layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),  
        layers.UpSampling2D((2, 2)),  
        layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')  
    ])  
      
    autoencoder = models.Model(encoder.input, decoder(encoder.output))  
    autoencoder.compile(optimizer='adam', loss='mse')  
      
    return autoencoder, encoder  