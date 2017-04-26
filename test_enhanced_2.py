import pandas as pd
import numpy as np
import os
import pre_process_last
import cnn_model_batch_norm
from skimage import io

test = pd.read_csv('GT-final_test.csv',sep=';')

model = cnn_model_batch_norm.cnn_model()

model.load_weights('enhance_2_model.h5')
# Load test dataset
X_test = []
y_test = []
i = 0
for file_name, class_id  in zip(list(test['Filename']), list(test['ClassId'])):
    img_path = os.path.join('GTSRB/Final_Test/Images/',file_name)
    X_test.append(pre_process_last.preprocess_img(io.imread(img_path)))
    y_test.append(class_id)
    
X_test = np.array(X_test)
y_test = np.array(y_test)

# predict and evaluate
y_pred = model.predict_classes(X_test)
acc = np.sum(y_pred==y_test)*1.0/np.size(y_pred)
print("Test accuracy = {}".format(acc))
