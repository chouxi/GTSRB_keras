import pandas as pd
import numpy as np
import os
import pre_process_last
import vgg16_model
from skimage import io
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout

test = pd.read_csv('GT-final_test.csv',sep=';')

model = vgg16_model.vgg16_model([pre_process_last.IMG_SIZE, pre_process_last.IMG_SIZE, 3], pre_process_last.NUM_CLASSES)

top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(pre_process_last.NUM_CLASSES, activation='softmax'))
model.add(top_model)

model.load_weights('enhance_1_model.h5')
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
