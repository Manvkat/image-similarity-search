from sklearn.metrics import precision_score, recall_score, accuracy_score  
  
def evaluate_performance(y_true, y_pred):  
    precision = precision_score(y_true, y_pred, average='weighted')  
    recall = recall_score(y_true, y_pred, average='weighted')  
    accuracy = accuracy_score(y_true, y_pred)  
    return precision, recall, accuracy  