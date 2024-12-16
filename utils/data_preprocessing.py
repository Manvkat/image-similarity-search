import os  
import cv2  
import numpy as np  
from sklearn.model_selection import train_test_split  
  
def load_and_preprocess_data(data_dir, img_size=(32, 32)):  
    images = []  
    labels = []  
    label_map = {label: idx for idx, label in enumerate(os.listdir(data_dir))}  
      
    for label in os.listdir(data_dir):  
        for img_file in os.listdir(os.path.join(data_dir, label)):  
            img_path = os.path.join(data_dir, label, img_file)  
            img = cv2.imread(img_path)  
            img = cv2.resize(img, img_size)  
            images.append(img)  
            labels.append(label_map[label])  
      
    images = np.array(images)  
    labels = np.array(labels)  
    images = images / 255.0  # Normalize the images  
      
    return train_test_split(images, labels, test_size=0.2, random_state=42)  