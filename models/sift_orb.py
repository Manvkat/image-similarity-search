import cv2  
import numpy as np  
  
def extract_sift_features(image):  
    sift = cv2.SIFT_create()  
    keypoints, descriptors = sift.detectAndCompute(image, None)  
    return keypoints, descriptors  
  
def extract_orb_features(image):  
    orb = cv2.ORB_create()  
    keypoints, descriptors = orb.detectAndCompute(image, None)  
    return keypoints, descriptors  